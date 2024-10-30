# Diffuser Mujoco

## Installation

To install the package in editable mode, run:
### Install pytorch
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
### Install Mujoco
https://github.com/openai/mujoco-py#install-mujoco

### Install D4RL
```bash
cd <PATH_TO_D4RL_INSTALL_DIR>
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
pip install -e .
```
### Download VD4RL
Please aware that V-d4rl is a bias dataset.
we use 64*64 image-based dataset

https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI

or

```bash
wget 'https://drive.usercontent.google.com/download?id=1XlVp6_pIISdv9PszPui6afrAoF_K3q-7&export=download&authuser=0&confirm=t&uuid=a0955024-3b12-496f-a032-a0b851e850d5&at=AN_67v0_dl9jG5c6tKvqrcx9ZREt:1727316143094' -O vd4rl.zip
```
### Download Robomimic
```bash
# can 
pip install gdown
gdown https://drive.google.com/uc?id=1q1X5muyM1bibFOEbpAU8YeOSNx4W6Baq

```
### Install
```bash
cd PATH_TO_DIFFUSER_MUJOCO
pip install -e .
```


## Training

To start training, execute the following script:

```bash
python scripts/train.py
```