import json
import math
from dataclasses import dataclass
from os import PathLike, cpu_count
from pathlib import Path
from typing import Optional, TypeAlias

import colorcet as cc
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rich.traceback import install as traceback_install
from simple_parsing import field, flag, parse_known_args
from timm.data import create_transform, resolve_data_config
from timm.models import ConvNeXt, SwinTransformer, VisionTransformer
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

_ = traceback_install(show_locals=True, locals_max_length=0)

# Type aliases
TaggerModel: TypeAlias = VisionTransformer | SwinTransformer | ConvNeXt

# working dir, either file parent dir or cwd if interactive
work_dir = (Path(__file__).parent if "__file__" in locals() else Path.cwd()).resolve()
# pick torch device based on GPU availability
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# map model variants to their HF model repo id
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}
# allowed input extensions
IMAGE_EXTNS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]


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


def pil_make_grid(
    images: list[Image.Image],
    max_cols: int = 5,
    padding: int = 2,
    bg_color: tuple[int, int, int] = (40, 42, 54),  # dracula background color
) -> Image.Image:
    n_cols = min(math.floor(math.sqrt(len(images))), max_cols)
    n_rows = math.ceil(len(images) / n_cols)

    # assumes all images are same size
    image_width, image_height = images[0].size

    canvas_width = ((image_width + padding) * n_cols) + padding
    canvas_height = ((image_height + padding) * n_rows) + padding

    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    for i, img in enumerate(images):
        x = (i % n_cols) * (image_width + padding) + padding
        y = (i // n_cols) * (image_height + padding) + padding
        canvas.paste(img, (x, y))

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
    caption = ", ".join(combined_names).replace("(", "\(").replace(")", "\)")
    taglist = caption.replace("_", " ")

    return caption, taglist, rating_labels, char_labels, gen_labels


@torch.no_grad()
def render_heatmap(
    image: Tensor,
    gradients: Tensor,
    image_feats: Tensor,
    image_probs: Tensor,
    image_labels: list[str],
    image_path: Path,
    output_dir: Path,
    cmap: LinearSegmentedColormap = cc.m_linear_bmy_10_95_c71,
    pos_embed_dim: int = 784,
    image_size: tuple[int, int] = (448, 448),
    font_args: dict = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1,
        "color": (255, 255, 255),
        "thickness": 2,
        "lineType": cv2.LINE_AA,
    },
) -> tuple[list[tuple[float, str, Image.Image]], Image.Image]:
    hmap_dim = int(math.sqrt(pos_embed_dim))

    image_hmaps = gradients.mean(2, keepdim=True).mul(image_feats.unsqueeze(0)).squeeze()
    image_hmaps = image_hmaps.mean(-1).reshape(len(image_labels), hmap_dim, hmap_dim)
    image_hmaps = image_hmaps.max(torch.zeros_like(image_hmaps))

    image_hmaps /= image_hmaps.reshape(image_hmaps.shape[0], -1).max(-1)[0].unsqueeze(-1).unsqueeze(-1)
    # normalize to 0-1
    image_hmaps = torch.stack([(x - x.min()) / (x.max() - x.min()) for x in image_hmaps]).unsqueeze(1)
    # interpolate to input image size
    image_hmaps = F.interpolate(image_hmaps, size=image_size, mode="bilinear").squeeze(1)

    hmap_imgs = []
    for tag, hmap, score in zip(image_labels, image_hmaps, image_probs.cpu()):
        image_pixels = image.add(1).mul(127.5).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        hmap_pixels = cmap(hmap.cpu().numpy(), bytes=True)[:, :, :3]

        hmap_cv2 = cv2.cvtColor(hmap_pixels, cv2.COLOR_RGB2BGR)
        hmap_image = cv2.addWeighted(image_pixels, 0.5, hmap_cv2, 0.5, 0)
        if tag is not None:
            cv2.putText(hmap_image, tag, (10, 30), **font_args)
            cv2.putText(hmap_image, f"{score:.3f}", org=(10, 60), **font_args)

        hmap_pil = Image.fromarray(cv2.cvtColor(hmap_image, cv2.COLOR_BGR2RGB))
        hmap_imgs.append((score, tag, hmap_pil))

    hmap_imgs = sorted(hmap_imgs, key=lambda x: x[0], reverse=True)
    hmap_grid = pil_make_grid([x[-1] for x in hmap_imgs], max_cols=6, padding=4)

    return hmap_imgs, hmap_grid


def process_batch_heatmaps(
    model: TaggerModel,
    labels: LabelData,
    images: Tensor,
    paths: list[Path],
    output_dir: Path,
    threshold: float = 0.5,
    suffix: str = ".txt",
):
    images_probs: list[Tensor] = []
    images_labels: list[str] = []
    images_grads: list[Tensor] = []

    with torch.set_grad_enabled(True):
        features = model.forward_features(images.to(torch_device))
        probs = model.forward_head(features)
        probs = F.sigmoid(probs)

        for idx in range(probs.shape[0]):
            probs_mask = probs[idx] > threshold
            probs_filtered = probs[idx][probs_mask]
            images_probs.append(probs_filtered)

            label_indices = torch.nonzero(probs_mask, as_tuple=False).squeeze(1)

            image_labels = [labels.names[label_indices[i]] for i in range(len(label_indices))]
            images_labels.append(image_labels)

        for idx, (image_probs, image_labels) in enumerate(zip(images_probs, images_labels)):
            eye = torch.eye(image_probs.shape[0], device=torch_device)
            grads = torch.autograd.grad(
                outputs=image_probs,
                inputs=features,
                grad_outputs=eye,
                is_grads_batched=True,
                retain_graph=True,
            )
            # yeah this means i end up doing backward for the same batch multiple times. idk how to avoid that
            image_grads = grads[0].detach().requires_grad_(False)[:, idx, :, :].unsqueeze(1)
            images_grads.append(image_grads)

    with torch.set_grad_enabled(False):
        for idx, (image, grads) in enumerate(zip(images, images_grads)):
            image_path = paths[idx]

            hmap_dir = output_dir.joinpath(image_path.stem.rstrip(" _-"))
            hmap_dir.mkdir(exist_ok=True, parents=True)

            hmap_imgs, hmap_grid = render_heatmap(
                image=image,
                gradients=grads,
                image_feats=features[idx],
                image_probs=images_probs[idx],
                image_labels=images_labels[idx],
                image_path=image_path,
                output_dir=output_dir,
            )
            for score, tag, hmap_pil in hmap_imgs:
                hmap_pil.save(hmap_dir.joinpath(f"{image_path.stem}_{score:0.3f}_{tag}.png"))
            hmap_grid.save(hmap_dir.joinpath(f"{image_path.stem}_heatmaps.png"))

            caption, taglist, ratings, character, general = get_tags(
                probs=probs[idx].cpu(),
                labels=labels,
                gen_threshold=threshold,
                char_threshold=threshold,
            )
            meta_path = hmap_dir.joinpath(image_path.stem + "_tags" + suffix)
            meta_path.write_text(taglist)
            meta_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "caption": caption,
                        "tags": taglist.split(", "),
                        "rating": ratings,
                        "general": general,
                        "character": character,
                    },
                    indent=4,
                    default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
                )
            )


@torch.inference_mode()
def run_tagger(
    model: TaggerModel,
    labels: LabelData,
    images: Tensor,
    paths: list[Path],
    output_dir: Path,
    suffix: str = ".txt",
    gen_threshold: float = 0.5,
    char_threshold: float = 0.85,
):
    features = model.forward_features(images.to(torch_device))
    probs = model.forward_head(features)
    probs = F.sigmoid(probs).cpu()

    for idx in range(probs.shape[0]):
        caption, taglist, ratings, character, general = get_tags(
            probs=probs[idx].cpu(),
            labels=labels,
            gen_threshold=gen_threshold,
            char_threshold=char_threshold,
        )
        meta_path = output_dir.joinpath(paths[idx].stem + "_tags" + suffix)
        meta_path.write_text(taglist)
        meta_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "caption": caption,
                    "tags": taglist.split(", "),
                    "rating": ratings,
                    "general": general,
                    "character": character,
                },
                indent=4,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
            )
        )


class ImageLoader(Dataset):
    def __init__(
        self,
        images_path: PathLike,
        transform: T.Compose,
    ):
        if not images_path.is_dir():
            raise FileNotFoundError(f"Image directory {images_path} does not exist or is not a directory.")
        self.images_path = Path(images_path)
        self.transform = transform
        self._preload()

    def _preload(self):
        self.image_files = [x for x in self.images_path.rglob("**/*") if x.suffix.lower() in IMAGE_EXTNS]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int | slice) -> dict[str, Tensor]:
        img_path = self.image_files[idx]
        image = Image.open(img_path)
        image = pil_ensure_rgb(image)
        image = self.transform(image)
        # NCHW convert RGB to BGR (cursed openCV legacy nonsense)
        image = image[[2, 1, 0]]
        return {"image": image, "path": np.bytes_(str(img_path))}


@dataclass
class ScriptOptions:
    images_path: Path = field(positional=True)

    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)

    tag_suffix: str = field(default=".txt")
    batch_size: int = field(default=1)
    num_workers: Optional[int] = field(default=None)

    gradcam: bool = flag(default=False)
    output_dir: Optional[Path] = field(default=None)


def main(opts: ScriptOptions):
    repo_id = MODEL_REPO_MAP.get(opts.model)
    images_path = Path(opts.images_path).resolve()
    output_dir = Path(opts.output_dir).resolve() if opts.output_dir else None

    if opts.gradcam is True and opts.model != "vit":
        raise ValueError("GradCAM heatmaps are only supported for the ViT model")

    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    if "model" not in locals():
        model = timm.create_model("hf-hub:" + repo_id, pretrained=True).eval()
    model = model.to(torch_device)

    print("Loading model's tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Creating input image transform...")
    model.pretrained_cfg["crop_mode"] = "border"
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # handle images path
    if images_path.is_dir():
        # multi-file mode
        print("Running in multi-image mode, preparing dataloader")
        multi_file = True
        # set output dir to ./heatmaps if using ./images, else to {images_path}/heatmaps
        if images_path.resolve() == work_dir.joinpath("images").resolve():
            # using the images folder so lets use the heatmaps one
            if opts.output_dir is None:
                output_dir = work_dir.joinpath("heatmaps")
            else:
                output_dir = images_path.joinpath("heatmaps")

        # load the dataset
        dataset = ImageLoader(images_path=images_path, transform=transform)

        # work out how many workers to use
        if opts.num_workers is not None:
            num_workers = opts.num_workers
        else:
            num_workers = cpu_count() if len(dataset) > 1000 else 0
        print(f"Using {num_workers} dataloader workers")

        # create the dataloader
        batch_size = min(opts.batch_size, len(dataset))
        inputs = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers or 0,
            shuffle=False,
        )

    elif images_path.is_file():
        # single file mode
        print("Running in single-image mode, preprocessing image")
        multi_file = False
        # set the heatmap base dir dir if unset
        output_dir = output_dir or images_path.parent.joinpath("heatmaps")
        # get image
        img_input: Image.Image = Image.open(images_path)
        # ensure image is RGB
        img_input = pil_ensure_rgb(img_input)
        # run the model's input transform to convert to tensor and rescale
        img_tensor = transform(img_input).unsqueeze(0)
        # NCHW image RGB to BGR (cursed openCV legacy nonsense)
        inputs = img_tensor[:, [2, 1, 0]]
    else:
        raise FileNotFoundError(f"Image file/folder '{images_path}' does not exist!")

    print(f"Will save heatmaps and tags in {output_dir}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    if multi_file:
        print("Starting batch processing...")
        for batch in tqdm(inputs, desc="Processing batch", unit="image", unit_scale=batch_size):
            batch_paths = [Path(x.decode("utf-8")) for x in batch["path"]]
            batch_images = batch["image"]

            if opts.gradcam:
                process_batch_heatmaps(
                    model=model,
                    labels=labels,
                    images=batch_images,
                    paths=batch_paths,
                    output_dir=output_dir,
                    threshold=min(opts.gen_threshold, opts.char_threshold),
                    suffix=opts.tag_suffix,
                )
            else:
                run_tagger(
                    model=model,
                    labels=labels,
                    images=batch_images,
                    paths=batch_paths,
                    output_dir=output_dir,
                    suffix=opts.tag_suffix,
                    gen_threshold=opts.gen_threshold,
                    char_threshold=opts.char_threshold,
                )
    else:
        print("Processing single image...")
        if opts.gradcam:
            process_batch_heatmaps(
                model=model,
                labels=labels,
                images=inputs,
                paths=[images_path],
                output_dir=output_dir,
                threshold=min(opts.gen_threshold, opts.char_threshold),
                suffix=opts.tag_suffix,
            )
        else:
            run_tagger(
                model=model,
                labels=labels,
                images=inputs,
                paths=[images_path],
                output_dir=output_dir,
                suffix=opts.tag_suffix,
                gen_threshold=opts.gen_threshold,
                char_threshold=opts.char_threshold,
            )


if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
