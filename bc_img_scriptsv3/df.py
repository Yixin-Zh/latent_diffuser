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


from diffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from diffuser.env.async_vector_env import AsyncVectorEnv
from diffuser.env.utils import VideoRecorder
from diffuser.dataset.robomimic_datasetv2 import RobomimicDataset

from diffuser.bc.bc_img_agentv3 import Agent
from diffuser.diffusion import ContinuousDiffusionSDE

from diffuser.nn_diffusion import DiT1d

from diffuser.dataset.dataset_utils import loop_dataloader
from diffuser.utils import report_parameters
from diffuser.utils.logger import Logger



@hydra.main(config_path=".", config_name="df.yaml", version_base=None)
def pipeline(args):
    # ---------------------- Create Save Path ----------------------
    save_path = f'results/{args.project}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # ---------------------- Create Single Logger ----------------------
    logger = Logger(pathlib.Path(save_path), args)

    # ---------------------- Create World Model ----------------------
    agent = Agent(latent_dim= 64, action_dim=7, device=args.device)
    agent.load(args.agent_model_path)
    agent.eval()


    logger = Logger(pathlib.Path(save_path), args)
    
    # ---------------------- Start Training Diffuser ----------------------
    
    dataset = RobomimicDataset(dataset_dir= args.datapath, shape_meta= args.shape_meta,
                                sequence_length=8,
                                abs_action=args.abs_action,)
    dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    # --------------- Network Architecture -----------------
    nn_diffusion = DiT1d(
        64+7, 128,
        d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")
    
     # ----------------- Masking -------------------
    fix_mask = torch.zeros((8, 64 + 7))
    fix_mask[0, :64] = 1.
    loss_weight = torch.ones((8, 64 + 7))
    loss_weight[0, 64:] = 5.
    
    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    diffuser = ContinuousDiffusionSDE(
        nn_diffusion,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
        device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")
    
   
    
    # ---------------------- Training ----------------------
    diffusion_lr_scheduler = CosineAnnealingLR(diffuser.optimizer, args.diffusion_gradient_steps)
   
    
    diffuser.train()
   
    
    n_gradient_step = 0
    log = {"avg_loss_diffusion": 0., }
    n_gradient_step = 0
    for batch in loop_dataloader(dataloader):
        obs = batch['agentview_image'].to(args.device)
        act = batch['action'].to(args.device)
        latent_obs = agent.sample(obs)
    
        # ----------- Gradient Step ------------
        
        log["avg_loss_diffusion"] += diffuser.update(torch.cat((latent_obs, act), dim=-1), step=n_gradient_step)['loss']
        
        diffusion_lr_scheduler.step()
       
        # ----------- Logging ------------
        if (n_gradient_step + 1) % args.log_freq == 0:
            metrics = {
                'step': n_gradient_step,
                'avg_loss_diffusion': log["avg_loss_diffusion"] / (args.log_freq/200),
            }
            logger.log(metrics, 'train')  
            log = {k: 0. for k in log}
    
        # ----------- Saving ------------
        if (n_gradient_step + 1) % args.save_freq == 0:
            diffuser.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
            diffuser.save(save_path + f"diffusion_ckpt_latest.pt")
    
        n_gradient_step += 1
        if n_gradient_step >= args.diffusion_gradient_steps:
            break
    
    # ---------------------- Save Models and Finish Logging ----------------------
    logger.save_agent(diffuser, identifier='final')
    logger.finish()

if __name__ == "__main__":
    pipeline()
