import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffuser.nn.mlp import build_mlp

from diffuser.nn.encoder import ResNet18Enc 
from diffuser.nn.decoder import ResNet18DecV2

class Agent:

    def __init__(self, latent_dim, action_dim, device="cpu", optim_params=None):
        self.encoder = ResNet18Enc(z_dim=latent_dim)
        self.decoder = ResNet18DecV2(z_dim=latent_dim)
        self.inv = build_mlp(latent_dim * 2, [512, 512], action_dim, output_activation=None, use_layernorm=True)
        self.fwd = build_mlp(latent_dim + action_dim, [512, 512], latent_dim, output_activation=None, use_layernorm=True)

        self.model = nn.ModuleDict({
            "encoder": self.encoder,
            "decoder": self.decoder,
            "inv": self.inv,
            "fwd": self.fwd,
        }).to(device)

        if optim_params is None:
            optim_params = {"lr": 1e-4, "weight_decay": 1e-6}
        self.optimizer = optim.AdamW(self.model.parameters(), **optim_params)


    def loss(self, pred_act, act, pred_next_latent_obs, latent_next_obs, rec_obs, obs,):
  
        inv_loss = F.mse_loss(pred_act, act)
        fwd_loss = F.mse_loss(pred_next_latent_obs, latent_next_obs)
        rec_obs_loss = F.mse_loss(rec_obs, obs)
       

        return inv_loss, fwd_loss, rec_obs_loss, 
        
    def update(self, obs, next_obs, act, step):
        latent_obs = self.model["encoder"](obs)
        latent_next_obs = self.model["encoder"](next_obs)

        pred_act = self.model["inv"](torch.cat([latent_obs, latent_next_obs], dim=-1))
        pred_next_latent_obs = self.model["fwd"](torch.cat([latent_obs, act], dim=-1))
        next_latent_obs_label = latent_next_obs.clone().detach()

        rec_obs = self.model["decoder"](latent_obs)
        
        # Loss
        inv_loss, fwd_loss, rec_obs_loss= self.loss(pred_act, act, pred_next_latent_obs, next_latent_obs_label, rec_obs, obs,)
        loss = inv_loss + fwd_loss + rec_obs_loss 
    
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % 200 == 0:
            return {
                "step": step,
                "loss": loss.item(),
                "inv_loss": inv_loss.item(),
                "fwd_loss": fwd_loss.item(),
                "rec_obs_loss": rec_obs_loss.item(),
                }
        else:
            return {}

    def test(self, obs, next_obs, act, step):
        latent_obs = self.model["encoder"](obs)
        latent_next_obs = self.model["encoder"](next_obs)

        pred_act = self.model["inv"](torch.cat([latent_obs, latent_next_obs], dim=-1))
        pred_next_latent_obs = self.model["fwd"](torch.cat([latent_obs, act], dim=-1))
        next_latent_obs_label = latent_next_obs.clone().detach()

        rec_obs = self.model["decoder"](latent_obs)
        
        return latent_obs, latent_next_obs
        
    # def infer(self, obs, ):
    #     '''
    #     One-step inference
    #     Args:
    #         obs: torch.Tensor, shape (B, C, H, W)
    #     Returns:
    #         action: torch.Tensor, shape (B, action_dim)
    #     '''
    #     with torch.no_grad():
    #         latent_obs = self.model["encoder"](obs)
    #         pred_next_latent_obs = self.model["fwd"](torch.cat([latent_obs, act], dim=-1
    def sample(self, obs):
        '''
        Sample latent obs from obs
        Args:
            obs: torch.Tensor, shape (B, T, C, H, W)
        Returns:
            latent_obs: torch.Tensor, shape (B, T, latent_dim)
        '''
        with torch.no_grad():
            B, T, C, H, W = obs.shape
            obs = obs.reshape(-1, C, H, W)

            latent_obs = self.model["encoder"](obs)
            latent_obs = latent_obs.reshape(B, T, -1)

        return latent_obs
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def eval(self):
        self.model.eval()


