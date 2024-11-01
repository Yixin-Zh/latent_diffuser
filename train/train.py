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
from latent_diffuser.util import set_seed, parse_cfg
from torch.optim.lr_scheduler import CosineAnnealingLR

from latent_diffuser.dataset.robomimic_datasetv2 import RobomimicDataset

from latent_diffuser.dataset.dataset_utils import loop_dataloader
from latent_diffuser.utils import report_parameters
from latent_diffuser.utils.logger import Logger

from latent_diffuser.latent_diffuser import VAE, InvPolicy, Planner

@hydra.main(config_path=".", config_name="train.yaml", version_base=None)
def pipeline(args):
    # ---------------------- Create Save Path ----------------------
    save_path = f'results/{args.project}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # ---------------------- Create Single Logger ----------------------
    logger = Logger(pathlib.Path(save_path), args)

    # ---------------------- Create VAE ----------------------
    vae1 = VAE(device=args.device) # for agent view
    vae2 = VAE(device=args.device) # for robot eye in hand view

    # ---------------------- Create Inverse Dynamics Diffsusion Policy ----------------------
    invdyn = InvPolicy(device=args.device)
    invdyn.train()
    # ---------------------- Create Latent Planner ----------------------
    planner = Planner(horizon=args.horizon, device=args.device)
    planner.train()
    logger = Logger(pathlib.Path(save_path), args)
    
    # ---------------------- Start Training Diffuser ----------------------
    dataset = RobomimicDataset(dataset_dir=args.datapath, shape_meta= args.shape_meta,
                                sequence_length=args.horizon,
                                abs_action=args.abs_action,)
    dataloader = DataLoader(dataset, 32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    

    n_gradient_step = 0
    # typical setting from the paper
    vae_gradient_steps = 50000
    invdyn_gradient_steps = vae_gradient_steps + 500000
    planner_gradient_steps = invdyn_gradient_steps + 1000000

    # vae_gradient_steps = 10
    # invdyn_gradient_steps = 20
    # planner_gradient_steps = 30
    for batch in loop_dataloader(dataloader):
        
        obs1 = batch['agentview_image'].to(args.device)
        obs2 = batch['robot0_eye_in_hand_image'].to(args.device)
        action = batch['action'].to(args.device)

        # ----------- VAE Update ------------
        if n_gradient_step < vae_gradient_steps:
            B, T, C, H, W = obs1.shape
            obs1 = obs1.view(B*T, C, H, W)
            obs2 = obs2.view(B*T, C, H, W)
            log = vae1.update(obs1, step=n_gradient_step)
            vae2.update(obs2, step=n_gradient_step)
        
        # make vae to eval mode once it is trained
        if n_gradient_step == vae_gradient_steps:
            vae1.eval()
            vae2.eval()
        
        # ----------- Inverse Dynamics Update ------------
        if n_gradient_step >= vae_gradient_steps and n_gradient_step < invdyn_gradient_steps:
            B, T, C, H, W = obs1.shape
            obs1 = obs1.view(B*T, C, H, W)
            obs2 = obs2.view(B*T, C, H, W)

            latent_obs1 = vae1.sample_latent(obs1) # shape: (B*T, C//2, H, W)
            latent_obs2 = vae2.sample_latent(obs2)
            
            latent_obs1 = latent_obs1.view(B, T, -1) # shape: (B, T, dim)
            latent_obs2 = latent_obs2.view(B, T, -1) # shape: (B, T, dim)
            latent = torch.cat([latent_obs1, latent_obs2], dim=-1) # shape: (B, T, dim*2)

            cur_latent = latent[:, :-1] # shape: (B, T-1, dim*2)
            next_latent = latent[:, 1:] # shape: (B, T-1, dim*2)

            act = action[:, :-1] # shape: (B, T-1, 7)

            latent = torch.cat([cur_latent, next_latent], dim=-1).reshape(B*(T-1), -1) # shape: (B*(T-1), dim*4)
            act = act.reshape(B*(T-1), -1).unsqueeze(1) # shape: (B*(T-1), 1, 7)
            
            log = invdyn.update(latent_obs=latent, act=act, step=n_gradient_step)
        
        if n_gradient_step == invdyn_gradient_steps:
            invdyn.eval()
    
        # ----------- Planner Update ------------
        if n_gradient_step >= invdyn_gradient_steps and n_gradient_step < planner_gradient_steps:
            B, T, C, H, W = obs1.shape
            obs1 = obs1.view(B*T, C, H, W)
            obs2 = obs2.view(B*T, C, H, W)

            latent_obs1 = vae1.sample_latent(obs1) # shape: (B*T, C//2, H, W)
            latent_obs2 = vae2.sample_latent(obs2)

            latent_obs1 = latent_obs1.view(B, T, -1) # shape: (B, T, C//2*H*W)
            latent_obs2 = latent_obs2.view(B, T, -1) # shape: (B, T, C//2*H*W)

            latent = torch.cat([latent_obs1, latent_obs2], dim=-1) # shape: (B, T, C*H*W)
            log = planner.update(latent_obs=latent, step=n_gradient_step)
       
        # ----------- Logging ------------
        if (n_gradient_step + 1) % args.log_freq == 0:
            
            logger.log(log, 'train')  
            
    
        # ----------- Saving ------------
        if (n_gradient_step + 1) % args.save_freq == 0:
            if n_gradient_step < vae_gradient_steps:
                vae1.save(save_path + f"vae1_ckpt_{n_gradient_step}.pt")
                vae2.save(save_path + f"vae2_ckpt_{n_gradient_step}.pt")
            elif n_gradient_step >= vae_gradient_steps and n_gradient_step < invdyn_gradient_steps:
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step}.pt")
            elif n_gradient_step >= invdyn_gradient_steps and n_gradient_step < planner_gradient_steps:
                planner.save(save_path + f"planner_ckpt_{n_gradient_step}.pt")    
            
    
        n_gradient_step += 1
        print(f"Gradient Step: {n_gradient_step}")
        if n_gradient_step >= planner_gradient_steps:
            # save everything and break
            vae1.save(save_path + f"vae1_ckpt_latest.pt")
            vae2.save(save_path + f"vae2_ckpt_latest.pt")
            invdyn.save(save_path + f"invdyn_ckpt_latest.pt")
            planner.save(save_path + f"planner_ckpt_latest.pt")
            break
    
    # ---------------------- Finish Logging ----------------------
    logger.finish()

if __name__ == "__main__":
    pipeline()
