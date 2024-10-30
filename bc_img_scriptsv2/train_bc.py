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

@hydra.main(config_path=".", config_name="bc.yaml", version_base=None)
def pipeline(args):

    # ---------------------- Create Save Path ----------------------
    save_path = f'results/{args.project}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # ---------------------- Create Logger ----------------------
    logger = Logger(pathlib.Path(save_path), args)
    
    # ---------------------- Load Expert Data ----------------------
    datapath = args.datapath
    print(args.shape_meta)
    dataset = RobomimicDataset(dataset_dir=datapath, shape_meta= args.shape_meta,
                                sequence_length=8,
                                abs_action=args.abs_action,)
    dataloader = DataLoader(dataset, 8, shuffle=True, num_workers=8, pin_memory=True)

    # ---------------------- Create Agent ----------------------
    agent = Agent(latent_dim=256, action_dim=7, device=args.device)
    
    # ---------------------- Training ----------------------
    step = 0
    max_steps = 200000
    
    # torch.cuda.synchronize()
    for batch in loop_dataloader(dataloader):
        
       
        obs_ = batch['agentview_image'].to(args.device)
        action_ = batch['action'].to(args.device)
        obs = obs_[:, :-1].reshape(-1, *obs_.shape[2:])
        next_obs = obs_[:, 1:].reshape(-1, *obs_.shape[2:])

        act = action_[:, :-1].reshape(-1, *action_.shape[2:])
        
        log = agent.update(obs, next_obs, act, step)
        
        
        
        
        if step % 200 == 0:
            logger.log(log, 'train')
        if step % args.save_freq == 0:
            agent.save(os.path.join(save_path, f'model_{step}'))
            agent.save(os.path.join(save_path, 'model_latest'))
        step += 1
        if step >= max_steps:
            break
    # ---------------------- Save Final Model ----------------------
    agent.save(os.path.join(save_path, 'model_latest'))
    logger.finish()

if __name__ == "__main__":
    pipeline()