import hydra
import os
# os.environ['HYDRA_FULL_ERROR'] = '1'
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffuser.util import set_seed, parse_cfg
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffuser.dataset.robomimic_datasetv2 import RobomimicDataset

from diffuser.dataset.dataset_utils import loop_dataloader
from diffuser.utils import report_parameters
from diffuser.utils.logger import Logger

from diffuser.bc.bc_img_agengv2 import Agent
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_tsne(data, title, savepath):
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    latent_2d = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=np.arange(len(data)), cmap='viridis', marker='o', alpha=0.7)
    plt.colorbar(scatter, label='Data Point Index')

    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.savefig(savepath)
    plt.close()

@hydra.main(config_path=".", config_name="bc.yaml", version_base=None)
def pipeline(args):
    
   
    
    # ---------------------- Load Expert Data ----------------------
    datapath = args.datapath
    dataset = RobomimicDataset(dataset_dir=datapath, shape_meta= args.shape_meta,
                                sequence_length=1,
                                abs_action=args.abs_action,)
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # ---------------------- Create Agent ----------------------
    agent = Agent(latent_dim=256, action_dim=7, device=args.device)
    agent.load(args.model_path,)
    agent.eval()
    # ---------------------- Training ----------------------
    step = 0
    max_steps = 100
    latent_obs_list = []
   
    for batch in loop_dataloader(dataloader):
        import time
        t1 = time.time()
        obs = batch['agentview_image'].to(args.device) # (B, T, C, H, W)
        
        latent_obs = agent.sample(obs   )
        latent_obs_list.append(latent_obs.reshape(-1, latent_obs.shape[-1]))
        
        step += 1
        print(step)
        if step >= 300:
            break

    latent_obs = torch.cat(latent_obs_list, dim=0).cpu().detach().numpy()
    latent_obs = latent_obs.reshape(-1, latent_obs.shape[-1])
    print(latent_obs.shape)
   
    plot_tsne(latent_obs, 'latent space', 'latent_tsne.png')


    
if __name__ == "__main__":
    pipeline()