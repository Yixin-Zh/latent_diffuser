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

def linear_interpolation(latent_obs1, latent_obs2, num_steps):
    latent_obs1 = latent_obs1.reshape(-1, latent_obs1.shape[-1])
    latent_obs2 = latent_obs2.reshape(-1, latent_obs2.shape[-1])
    latent_obs_list = []
    latent_obs_list.append(latent_obs1)
    for i in range(num_steps):
        latent_obs = latent_obs1 + (latent_obs2 - latent_obs1) * i / num_steps
        latent_obs_list.append(latent_obs)
    latent_obs_list.append(latent_obs2)
    latent_obs = np.concatenate(latent_obs_list, axis=0)
    return latent_obs

def plot_img_seq_gif(seq, savepath):
    import imageio
    from PIL import Image
    import cv2
    seq = (seq * 255).astype(np.uint8)  # shape (T, C, H, W)
    seq = np.transpose(seq, (0, 2, 3, 1))  # shape (T, H, W, C)
    seq = np.clip(seq, 0, 255)
    # seq = seq[:, :, :, ::-1]  # BGR to RGB

    frames = [Image.fromarray(frame) for frame in seq]
    frames[0].save(savepath, save_all=True, append_images=frames[1:], duration=100, loop=0)

@hydra.main(config_path=".", config_name="bc.yaml", version_base=None)
def pipeline(args):
    
   
    
    # ---------------------- Load Expert Data ----------------------
    datapath = args.datapath
    dataset = RobomimicDataset(dataset_dir=datapath, shape_meta= args.shape_meta,
                                sequence_length=50,
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
        
        latent_obs = agent.sample(obs) # (B, T, feature_dim)
        latent_obs1 = latent_obs[0, 0]
        latent_obs2 = latent_obs[0, -1]
        latent_obs_seq = linear_interpolation(latent_obs1.cpu().numpy(), latent_obs2.cpu().numpy(), 18) # (T, feature_dim)
        latent_obs_seq = torch.tensor(latent_obs_seq).to(args.device)
        
        inter_rec_obs_seq = agent.model["decoder"](latent_obs_seq)
        inter_rec_obs_seq = inter_rec_obs_seq.cpu().detach().numpy()

        rec_obs_seq = agent.model["decoder"](latent_obs[0])
        rec_obs_seq = rec_obs_seq.cpu().detach().numpy()
        plot_img_seq_gif(obs[0].cpu().numpy(), f'obs_seq_{step}.gif')
        plot_img_seq_gif(rec_obs_seq, f'rec_img_seq_{step}.gif')
        plot_img_seq_gif(inter_rec_obs_seq, f'inter_img_seq_{step}.gif')

        step += 1
        print(step)
        if step >= 1:
            break
    # plot_tsne(latent_obs[0].cpu().numpy(), 'latent_obs', 'latent_obs_tsnexx.png')

    
if __name__ == "__main__":
    pipeline()