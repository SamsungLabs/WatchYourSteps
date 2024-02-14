# Watch Your Steps: Local Image and Scene Editing by Text Instructions

This repository contains an implementation of `Watch Your Steps`, a method for localized 2D image and 3D scene editing.
This project was done at the Samsung AI Centre in Toronto.

![teaser](imgs/wys-1.png)

<p align="center">
<b>Watch Your Steps: Local Image and Scene Editing by Text Instructions.</b><br>
Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, Igor Gilitschenski.
</p>
<p align="center">
  <a href="https://ashmrz.github.io/WatchYourSteps">Website</a> |
  <a href="https://arxiv.org/abs/2308.08947">ArXiv</a>
</p>

## Installation

### 2D Image Editing

Create a conda environment via

```
conda create --name wys2d
conda activate wys2d
python -m pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers torchtyping matplotlib accelerate
```

See also `egenv/environment-2d.yml` for an example environment.

### 3D Scene Editing

This aspect of the code builds upon [Nerfstudio](https://docs.nerf.studio/index.html) and [IN2N](https://github.com/ayaanzhaque/instruct-nerf2nerf).
Clone the github repo and `cd` into it. Then, run:

```
conda create --name wys -y python=3.8
conda activate wys
python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install --upgrade pip setuptools
pip install -e .
```

See also `egenv/environment-3d.yml` for an example environment.

## Editing 2D images

In `images-2d` we provide an implementation of WYS for 2D images, as well as an example usage script. Run:

```
cd images-2d
python example.py
```

See `images-2d/wys_2d.py` for a "drop-in" implementation of the 2D translation module.

## Editing 3D scenes

Like [IN2N](https://github.com/ayaanzhaque/instruct-nerf2nerf), we first train a standard NeRF (via NeRFStudio), followed by an editing phase.

We show how to do these in `ns-run.sh`.

## Citation

```
@article{mirzaei2023watch,
  title={Watch your steps: Local image and scene editing by text instructions},
  author={Mirzaei, Ashkan and Aumentado-Armstrong, Tristan and Brubaker, Marcus A and Kelly, Jonathan and Levinshtein, Alex and Derpanis, Konstantinos G and Gilitschenski, Igor},
  journal={arXiv preprint arXiv:2308.08947},
  year={2023}
}
```
