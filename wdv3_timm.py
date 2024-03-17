from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F
import cv2

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@dataclass
class ScriptOptions:
    image_file: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)


def main(opts: ScriptOptions):
    repo_id = MODEL_REPO_MAP.get(opts.model)
    image_path = Path(opts.image_file).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    model = timm.create_model("hf-hub:" + repo_id, pretrained=True).eval()

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Creating data transform...")
    model.pretrained_cfg['crop_mode'] = 'border'
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


    print("Loading image and preprocessing...")
    # get image
    img_input: Image.Image = Image.open(image_path)
    # ensure image is RGB
    img_input = pil_ensure_rgb(img_input)

    # run the model's input transform to convert to tensor and rescale
    inputs: Tensor = transform(img_input).unsqueeze(0)


    print("Running inference...")
    # with torch.inference_mode():
    # move model to GPU, if available
    if torch_device.type != "cpu":
        model = model.to(torch_device)
        inputs = inputs.to(torch_device)
    # run the model
    backbone_out =model.forward_features(inputs)
    outputs = model.forward_head(backbone_out)
    # apply the final activation function (timm doesn't support doing this internally)
    outputs = F.sigmoid(outputs)

    outputs_filtered_mask = outputs > 0.5
    #TODO: multiple images
    outputs_filtered = outputs[outputs_filtered_mask]
    outputs_filtered_idx = torch.nonzero(outputs_filtered_mask[0], as_tuple=False).squeeze(1)
    filtered_labels = [labels.names[outputs_filtered_idx[i]] for i in range(len(outputs_filtered_idx))]

    eye = torch.eye(outputs_filtered.shape[0], device=torch_device)
    gradients = torch.autograd.grad(outputs_filtered, backbone_out, is_grads_batched=True, grad_outputs=eye)
    heatmap = (gradients[0].mean(2, keepdim=True) * backbone_out.unsqueeze(0)).squeeze().mean(-1).reshape(len(filtered_labels),28,28)
    heatmap = heatmap.max(torch.zeros_like(heatmap))
    heatmap /= heatmap.reshape(heatmap.shape[0],-1).max(-1)[0].unsqueeze(-1).unsqueeze(-1)
    # heatmap =
    heatmap_dir = Path("heatmaps") / image_path.stem
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(filtered_labels)):
        heatmap[i] = (heatmap[i] - heatmap[i].min()) / (heatmap[i].max() - heatmap[i].min())
        cur_heatmap = F.interpolate(heatmap[i].unsqueeze(0).unsqueeze(0), size=(448,448), mode='bilinear').squeeze()
        jet_heatmap_cv2 = cv2.applyColorMap((cur_heatmap * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
        img_cv2 = cv2.cvtColor(((inputs/2+0.5)*255.0).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlayed_img_cv2 = cv2.addWeighted(img_cv2, 0.5, jet_heatmap_cv2, 0.5, 0)
        tag_name = filtered_labels[i]
        if tag_name is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)
            thickness = 2
            cv2.putText(overlayed_img_cv2, tag_name, (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(overlayed_img_cv2, f"{outputs_filtered[i]:.3f}", (10, 60), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.imwrite((heatmap_dir / f"{filtered_labels[i]}.png").as_posix(), overlayed_img_cv2)

    # move inputs, outputs, and model back to to cpu if we were on GPU
    if torch_device.type != "cpu":
        inputs = inputs.to("cpu")
        outputs = outputs.to("cpu").detach()

        model = model.to("cpu")



    print("Processing results...")
    caption, taglist, ratings, character, general = get_tags(
        probs=outputs.squeeze(0),
        labels=labels,
        gen_threshold=opts.gen_threshold,
        char_threshold=opts.char_threshold,
    )

    print("--------")
    print(f"Caption: {caption}")
    print("--------")
    print(f"Tags: {taglist}")

    print("--------")
    print("Ratings:")
    for k, v in ratings.items():
        print(f"  {k}: {v:.3f}")

    print("--------")
    print(f"Character tags (threshold={opts.char_threshold}):")
    for k, v in character.items():
        print(f"  {k}: {v:.3f}")

    print("--------")
    print(f"General tags (threshold={opts.gen_threshold}):")
    for k, v in general.items():
        print(f"  {k}: {v:.3f}")

    print("Done!")


if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
