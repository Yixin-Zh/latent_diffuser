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
    save_path = pathlib.Path(f'results/{args.project}/')
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(save_path, args)

    # ---------------------- Prepare Dataset ----------------------
    dataset = RobomimicDataset(
        dataset_dir=args.datapath,
        shape_meta=args.shape_meta,
        sequence_length=args.horizon,
        abs_action=args.abs_action,
    )

    # Define gradient steps for each phase
    vae_gradient_steps = 5
    invdyn_gradient_steps = vae_gradient_steps + 10
    planner_gradient_steps = invdyn_gradient_steps + 10

    # ---------------------- Phase 1: Train VAE1 and VAE2 ----------------------
    print("Starting Phase 1: Training VAE1 and VAE2")
    
    # Initialize VAE models
    vae1 = VAE(device=args.device)  # for agent view
    vae2 = VAE(device=args.device)  # for robot eye in hand view
    vae1.train()
    vae2.train()

    # Initialize DataLoader for VAE training
    vae_dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )

    vae_loader = loop_dataloader(vae_dataloader)
    for n_gradient_step in range(vae_gradient_steps):
        batch = next(vae_loader)

        obs1 = batch['agentview_image'].to(args.device)
        obs2 = batch['robot0_eye_in_hand_image'].to(args.device)
        B, T, C, H, W = obs1.shape
        obs1 = obs1.view(B * T, C, H, W)
        obs2 = obs2.view(B * T, C, H, W)

        # Update VAE models
        log_vae1 = vae1.update(obs1, step=int(n_gradient_step))
        log_vae2 = vae2.update(obs2, step=int(n_gradient_step))

        # Logging
        if current_step % 200 == 0:
            logger.log({**log_vae1,}, 'vae')

        # Saving checkpoints
        if (n_gradient_step + 1) % args.save_freq == 0:
            vae1.save(save_path / f"vae1_ckpt_{n_gradient_step + 1}.pt")
            vae2.save(save_path / f"vae2_ckpt_{n_gradient_step + 1}.pt")

        if (n_gradient_step + 1) % 1000 == 0:
            print(f"Phase 1 - Gradient Step: {n_gradient_step + 1}")

    # Save latest checkpoints
    vae1.save(save_path / "vae1_ckpt_latest.pt")
    vae2.save(save_path / "vae2_ckpt_latest.pt")
   

    # Set VAEs to eval mode
    vae1.eval()
    vae2.eval()

    # Delete VAE models to free CUDA memory
    del vae1
    del vae2
    torch.cuda.empty_cache()
    print("Phase 1 completed and CUDA memory freed.")

    # ---------------------- Phase 2: Train Inverse Dynamics Policy ----------------------
    print("Starting Phase 2: Training Inverse Dynamics Policy")

    # Initialize VAE models in eval mode for latent sampling
    vae1 = VAE(device=args.device)
    vae2 = VAE(device=args.device)
    vae1.load(save_path / "vae1_ckpt_latest.pt")
    vae2.load(save_path / "vae2_ckpt_latest.pt")
    vae1.eval()
    vae2.eval()

    # Initialize Inverse Dynamics Policy
    invdyn = InvPolicy(device=args.device)
    invdyn.train()

    # Initialize DataLoader for Inverse Dynamics training
    invdyn_dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )

    invdyn_loader = loop_dataloader(invdyn_dataloader)
    for n_gradient_step in range(vae_gradient_steps, invdyn_gradient_steps):
        batch = next(invdyn_loader)
        current_step = n_gradient_step  

        obs1 = batch['agentview_image'].to(args.device)
        obs2 = batch['robot0_eye_in_hand_image'].to(args.device)
        action = batch['action'].to(args.device)

        B, T, C, H, W = obs1.shape
        obs1 = obs1.view(B * T, C, H, W)
        obs2 = obs2.view(B * T, C, H, W)

        with torch.no_grad():
            latent_obs1 = vae1.sample_latent(obs1).view(B, T, -1)
            latent_obs2 = vae2.sample_latent(obs2).view(B, T, -1)
        
        latent = torch.cat([latent_obs1, latent_obs2], dim=-1)  # (B, T, dim*2)
        cur_latent = latent[:, :-1].reshape(B * (T - 1), -1)  # (B*(T-1), dim*2)
        next_latent = latent[:, 1:].reshape(B * (T - 1), -1)  # (B*(T-1), dim*2)
        action = action[:, :-1].reshape(B * (T - 1), -1).unsqueeze(1)  # (B*(T-1), 1, 7)

        combined_latent = torch.cat([cur_latent, next_latent], dim=-1)  # (B*(T-1), dim*4)

        # Update Inverse Dynamics Policy
        log_invdyn = invdyn.update(latent_obs=combined_latent, act=action, step=int(current_step))

        # Logging
        if current_step % 200 == 0:
            logger.log(log_invdyn, 'policy')

        # Saving checkpoints
        if (current_step + 1) % args.save_freq == 0:
            invdyn.save(save_path / f"invdyn_ckpt_{current_step + 1}.pt")

        if (current_step + 1) % 1000 == 0:
            print(f"Phase 2 - Gradient Step: {current_step + 1}")

    # Save latest checkpoint for Inverse Dynamics Policy
    invdyn.save(save_path / "invdyn_ckpt_latest.pt")
    

    # Set Inverse Dynamics Policy to eval mode
    invdyn.eval()

    # Delete Inverse Dynamics Policy and VAE models to free CUDA memory
    del invdyn
    del vae1
    del vae2
    torch.cuda.empty_cache()
    print("Phase 2 completed and CUDA memory freed.")

    # ---------------------- Phase 3: Train Planner ----------------------
    print("Starting Phase 3: Training Planner")

    # Initialize VAE models in eval mode for latent sampling
    vae1 = VAE(device=args.device)
    vae2 = VAE(device=args.device)
    vae1.load(save_path / "vae1_ckpt_latest.pt")
    vae2.load(save_path / "vae2_ckpt_latest.pt")
    vae1.eval()
    vae2.eval()

    # Initialize Planner
    planner = Planner(horizon=args.horizon, device=args.device)
    planner.train()

    # Initialize DataLoader for Planner training
    planner_dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )

    planner_loader = loop_dataloader(planner_dataloader)
    for n_gradient_step in range(invdyn_gradient_steps, planner_gradient_steps):
        batch = next(planner_loader)
        current_step = n_gradient_step  # Since n_gradient_step starts from 0

        obs1 = batch['agentview_image'].to(args.device)
        obs2 = batch['robot0_eye_in_hand_image'].to(args.device)

        B, T, C, H, W = obs1.shape
        obs1 = obs1.view(B * T, C, H, W)
        obs2 = obs2.view(B * T, C, H, W)

        with torch.no_grad():
            latent_obs1 = vae1.sample_latent(obs1).view(B, T, -1)
            latent_obs2 = vae2.sample_latent(obs2).view(B, T, -1)
        
        latent = torch.cat([latent_obs1, latent_obs2], dim=-1)  # (B, T, dim*2)

        # Update Planner
        log_planner = planner.update(latent_obs=latent, step=int(current_step))

        # Logging
        if current_step % 200 == 0:
            logger.log(log_planner, 'planner')

        # Saving checkpoints
        if (current_step + 1) % args.save_freq == 0:
            planner.save(save_path / f"planner_ckpt_{current_step + 1}.pt")

        if (current_step + 1) % 1000 == 0:
            print(f"Phase 3 - Gradient Step: {current_step + 1}")

    # Save latest checkpoint for Planner
    planner.save(save_path / "planner_ckpt_latest.pt")
 

    # Set Planner to eval mode
    planner.eval()

    # Delete Planner and VAE models to free CUDA memory
    del planner
    del vae1
    del vae2
    torch.cuda.empty_cache()
    print("Phase 3 completed and CUDA memory freed.")

    # ---------------------- Finish Logging ----------------------
    logger.finish()
    print("Training completed successfully.")

if __name__ == "__main__":
    pipeline()
