# Matching Textured Pointclouds with LVM


This code is based on [Project Webpage](https://diff3f.github.io/).

## Setup
```shell
conda env create -f environment.yaml
conda activate diff3f
```

### Additional prerequisites
[Install pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

You might face difficulty in installing pytorch3d or encounter the error `ModuleNotFoundError: No module named 'pytorch3d` during run time. Unfortunately, this is because pytorch3d could not be installed properly. Please refer [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for alternate ways to install pytorch3d. 

## Usage
Please check the example notebook [splats.ipynb] for details on computing features for a gaussian splats.


## BibTeX

The code is based on Diff3D code, please cite original work it as follows.

```bibtex
@article{dutt2023diffusion,
    title={Diffusion 3D Features (Diff3F): Decorating Untextured Shapes with Distilled Semantic Features}, 
    author={Dutt, Niladri Shekhar and Muralikrishnan, Sanjeev and Mitra, Niloy J.},
    journal={arXiv preprint arXiv:2311.17024},
    year={2023},
} 
``` 
