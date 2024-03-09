# wdv3-timm

small example thing showing how to use `timm` to run the WD Tagger V3 models.

doesn't seem to work right mind you but here we are

## How To Use

1. clone the repository and enter the directory:
```sh
git clone https://github.com/neggles/wdv3-timm.git
cd wd3-timm
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.10 -m venv .venv
# Activate it
source .venv/bin/activate
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install PyTorch manually (e.g. if you are not using an nVidia GPU)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install requirements
python -m pip install -r requirements.txt
```

3. Run the example script, picking one of the 3 models to use:
```sh
python wdv3_timm.py <swinv2|convnext|vit> path/to/image.png
```

Example output from `python wdv3_timm.py swinv2 test.png`:
```sh
Loading model 'swinv2' from 'SmilingWolf/wd-swinv2-tagger-v3'...
Loading tag list...
Creating data transform...
Loading image and preprocessing...
Running inference...
Processing results...
--------
Caption: 1girl, solo, animal_ears, bow, sleeves_past_wrists, jacket, scarf, brown_jacket, hair_bow, hairclip, animal_ear_fluff, open_jacket, blush, long_hair, open_clothes, hair_between_eyes, hand_up, purple_bow, long_sleeves, upper_body, shirt, white_shirt, black_jacket, hair_ornament, brown_cardigan, blazer, looking_at_viewer, closed_mouth, fringe_trim
--------
Tags: 1girl, solo, animal ears, bow, sleeves past wrists, jacket, scarf, brown jacket, hair bow, hairclip, animal ear fluff, open jacket, blush, long hair, open clothes, hair between eyes, hand up, purple bow, long sleeves, upper body, shirt, white shirt, black jacket, hair ornament, brown cardigan, blazer, looking at viewer, closed mouth, fringe trim
--------
Ratings:
  general: -0.334
  sensitive: 0.206
  questionable: -7.101
  explicit: -8.539
--------
Character tags (threshold=0.75):
--------
General tags (threshold=0.35):
  1girl: 6.164
  solo: 4.111
  animal_ears: 4.110
  bow: 3.570
  sleeves_past_wrists: 3.436
  jacket: 3.205
  scarf: 2.866
  brown_jacket: 2.596
  hair_bow: 2.095
  hairclip: 2.088
  animal_ear_fluff: 1.963
  open_jacket: 1.946
  blush: 1.767
  long_hair: 1.729
  open_clothes: 1.705
  hair_between_eyes: 1.500
  hand_up: 1.412
  purple_bow: 1.408
  long_sleeves: 1.383
  upper_body: 1.078
  shirt: 0.999
  white_shirt: 0.991
  black_jacket: 0.883
  hair_ornament: 0.826
  brown_cardigan: 0.723
  blazer: 0.676
  looking_at_viewer: 0.575
  closed_mouth: 0.538
  fringe_trim: 0.468
Done!

```
