# wdv3-timm

small example thing showing how to use `timm` to run the WD Tagger V3 models.

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
Loading model 'vit' from 'SmilingWolf/wd-vit-tagger-v3'...
Loading tag list...
Creating data transform...
Loading image and preprocessing...
Running inference...
Processing results...
--------
Caption: 1girl, horns, solo, bell, ahoge, colored_skin, blue_skin, neck_bell, looking_at_viewer, purple_eyes, upper_body, blonde_hair, long_hair, goat_horns, blue_hair, off_shoulder, sidelocks, bare_shoulders, alternate_costume, shirt, black_shirt, cowbell, ganyu_(genshin_impact)
--------
Tags: 1girl, horns, solo, bell, ahoge, colored skin, blue skin, neck bell, looking at viewer, purple eyes, upper body, blonde hair, long hair, goat horns, blue hair, off shoulder, sidelocks, bare shoulders, alternate costume, shirt, black shirt, cowbell, ganyu \(genshin impact\)
--------
Ratings:
  general: 0.827
  sensitive: 0.199
  questionable: 0.001
  explicit: 0.001
--------
Character tags (threshold=0.75):
  ganyu_(genshin_impact): 0.991
--------
General tags (threshold=0.35):
  1girl: 0.996
  horns: 0.950
  solo: 0.947
  bell: 0.918
  ahoge: 0.897
  colored_skin: 0.881
  blue_skin: 0.872
  neck_bell: 0.854
  looking_at_viewer: 0.817
  purple_eyes: 0.734
  upper_body: 0.615
  blonde_hair: 0.609
  long_hair: 0.607
  goat_horns: 0.524
  blue_hair: 0.496
  off_shoulder: 0.472
  sidelocks: 0.470
  bare_shoulders: 0.464
  alternate_costume: 0.437
  shirt: 0.427
  black_shirt: 0.417
  cowbell: 0.415
Done!
```
