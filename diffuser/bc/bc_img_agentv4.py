import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffuser.nn.mlp import build_mlp

from diffuser.nn.encoder import ResNet18Enc 
from diffuser.nn.decoder import ResNet18DecV3

class Agent:

    def __init__(self, latent_dim, action_dim, device="cpu", optim_params=None):
        self.encoder = ResNet18Enc(z_dim=latent_dim)
        self.decoder = ResNet18DecV3(action_dim=action_dim, latent_obs_dim=latent_dim)
        self.inv = build_mlp(latent_dim * 2, [1024, 1024, 512], action_dim, output_activation=None, use_layernorm=True)
        self.fwd = build_mlp(latent_dim + action_dim, [1024, 1024, 512], latent_dim, output_activation=None, use_layernorm=True)

        self.model = nn.ModuleDict({
            "encoder": self.encoder,
            "decoder": self.decoder,
            "inv": self.inv,
            "fwd": self.fwd,
        }).to(device)

        if optim_params is None:
            optim_params = {"lr": 1e-4, "weight_decay": 1e-6}
        self.optimizer = optim.AdamW(self.model.parameters(), **optim_params)


    def loss(self, pred_act, act, pred_next_latent_obs, latent_next_obs, rec_obs_diff, obs_diff, latent_obs):
  
        inv_loss = F.mse_loss(pred_act, act)
        rec_obs_diff_loss = F.mse_loss(rec_obs_diff, obs_diff)

        # Fwd loss: use cosine similarity 
        # No need to normalize the latent vectors 
        fwd_loss = (1 - F.cosine_similarity(pred_next_latent_obs, latent_next_obs, dim=-1)).mean()

        # add a loss to make latent_obs and pred_next_latent_obs to be different
        # this is to prevent the model from learning the identity function

        sim_loss = F.cosine_similarity(latent_obs, latent_next_obs, dim=-1).mean()

        return inv_loss, fwd_loss, rec_obs_diff_loss, sim_loss

    def batch_orthogonality_loss(self, features):
        """
        features: tensor of shape (B, T, C)
        """
        B, T, C = features.size()
        # Flatten the features to shape (B, T*C)
        features_flat = features.view(B, T * C)  # (B, T*C)
        
        # Normalize the feature vectors for each sample
        features_normalized = F.normalize(features_flat, p=2, dim=1)  # (B, T*C)
        
        # Compute the Gram matrix between samples in the batch (B, B)
        gram_matrix = torch.mm(features_normalized, features_normalized.t())  # (B, B)
        
        # Create an identity matrix (B, B)
        identity = torch.eye(B, device=features.device)
        
        # Compute the loss
        diff = gram_matrix - identity  # (B, B)
        loss = torch.sum(diff ** 2) / (B * B)
        
        return loss
    


    def update(self, obs, obs_diff, act, step):
        """
        obs: torch.Tensor, shape (B, T, C, H, W)
        obs_diff: torch.Tensor, shape (B, T-1, C, H, W)
        Note that obs_diff is the difference between obs[t+1] and obs[t]
        act: torch.Tensor, shape (B, T-1, action_dim)
        """

        B, T, C, H, W = obs.shape
        obs = obs.reshape(-1, C, H, W) # (B*T, C, H, W)
        obs_diff = obs_diff.reshape(-1, C, H, W) # (B*(T-1), C, H, W)
        act = act.reshape(-1, act.shape[-1]) # (B*(T-1), action_dim)

        # Encode
        latent_obs_ = self.model["encoder"](obs) # (B*T, latent_dim)
        latent_obs_ = latent_obs_.reshape(B, T, -1)

        latent_obs = latent_obs_[:, :-1].reshape(-1, latent_obs_.shape[-1]) # (B*(T-1), latent_dim)
        latent_next_obs = latent_obs_[:, 1:].reshape(-1, latent_obs_.shape[-1]) # (B*(T-1), latent_dim)

        # Fwd dynamics
        fwd_input = torch.cat([latent_obs, act], dim=-1)
        pred_next_latent_obs = self.model["fwd"](fwd_input)
        next_latent_obs_label = latent_next_obs.clone().detach()

        # Inv dynamics
        inv_input = torch.cat([latent_obs, latent_next_obs], dim=-1)
        pred_act = self.model["inv"](inv_input)

        # Decode
        rec_obs_diff = self.model["decoder"](act, latent_obs)

        inv_loss, fwd_loss, rec_obs_diff_loss, sim_loss = self.loss(pred_act, act, 
                        pred_next_latent_obs, next_latent_obs_label, 
                        rec_obs_diff, obs_diff,
                        latent_obs)
        
        # Orthogonality loss
        ortho_loss = self.batch_orthogonality_loss(latent_obs_.reshape(B, T, -1))

        # Compute loss
        loss = inv_loss + fwd_loss + rec_obs_diff_loss + (sim_loss + ortho_loss)* 0.1
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("Step: ", step, "Loss: ", loss.item(), "Ortho Loss: ", ortho_loss.item())
        if step % 200 == 0:
            return {
                "step": step,
                "loss": loss.item(),
                "inv_loss": inv_loss.item(),
                "fwd_loss": fwd_loss.item(),
                "rec_obs_diff_loss": rec_obs_diff_loss.item(),
                "sim_loss": sim_loss.item(),
                "ortho_loss": ortho_loss.item(),
                }
        else:
            return {}

        

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


