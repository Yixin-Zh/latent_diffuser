# Latent diffuser
Paper: https://openreview.net/pdf?id=k1qVBh5fnb
## Installation

To install the package in editable mode, run:
### Install pytorch
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
### Install Mujoco
https://github.com/openai/mujoco-py#install-mujoco

### Install
```bash
cd <PATH_TO_LATENT_DIFFUSER>
pip install -e .
```

### Install Robomimic/Robosuite
```bash
# Install Robomimic from source (recommended)
cd <PATH_TO_ROBOMIMIC_INSTALL_DIR>
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
cd <PATH_TO_ROBOSUITE_INSTALL_DIR>
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -e .
```
## Code Architecture
├── dataset
│   ├── base_dataset.py
│   ├── replay_buffer.py
│   ├── robomimic_dataset.py
│   ├── robomimic_datasetv2.py
│   └── rotation_conversions.py
│   └── ....
├── diffusion
│   ├── basic.py
│   ├── diffusionsde.py
├── latent_diffuser
│   ├── __init__.py
│   ├── invdyn.py # diffusion policy inverse dynamics
│   ├── planner.py # latent diffusion planner
│   └── vae.py # vae to get latent
├── nn
│   ├── __init__.py
│   ├── mlp.py
│   ├── module.py # encoder and decoder from stable diffusion
├── nn_diffusion
│   ├── base_nn_diffusion.py
│   ├── dit.py 
|.....

## Training

To start training, execute the following script:
You can change scripts/train.yaml, datapath to make your own datapath
All configs are here.
```bash
python scripts/train.py
```