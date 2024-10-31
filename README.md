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


### Install Robomimic/Robosuite

```bash
cd PATH_TO_LATENT_DIFFUSER
pip install -e .
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
### Install
```bash
cd <PATH_TO_LATENT_DIFFUSER>
pip install -e .
```
## Training

To start training, execute the following script:

```bash
python scripts/train.py
```