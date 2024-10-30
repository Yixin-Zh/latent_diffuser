from .mlp import build_mlp
from .decoder import ResNet18Dec
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionEncoder(nn.Module):
    def __init__(self, obs_dim, latent_act_dim):
        super(ActionEncoder, self).__init__()
        self.invdyn = build_mlp(
            input_dim= 2*obs_dim,
            hidden_dims=[256, 512, 256],
            output_dim=latent_act_dim,
            activation=nn.ReLU,
            output_activation=None,
            use_batchnorm=False
        )

    def forward(self, obs, next_obs):
        latent_act = self.invdyn(torch.cat([obs, next_obs], dim=-1))

        return latent_act



class ObsEncoder(nn.Module):
    def __init__(self, obs_dim, latent_obs_dim, act_dim):
        super(ObsEncoder, self).__init__()
        self.fordyn = build_mlp(
            input_dim= obs_dim + act_dim,
            hidden_dims=[256, 512, 256],
            output_dim=latent_obs_dim,
            activation=nn.ReLU,
            output_activation=None,
            use_batchnorm=False
        )

    def forward(self, obs, act):
       
        latent_obs = self.fordyn(torch.cat([obs, act], dim=-1))

        return latent_obs
    


class ObsDecoder(nn.Module):
    def __init__(self, obs_dim, latent_act_dim):
        super(ObsDecoder, self).__init__()
        self.fordyn = ResNet18Dec(
            obs_dim=obs_dim,
            latent_act_dim=latent_act_dim
        )

    def forward(self, obs, latent_act):
        """
        should be replaced with quantized version
        """
        next_obs = self.fordyn(obs, latent_act)

        return next_obs
    


class ActionDecoder(nn.Module):
    def __init__(self, obs_dim, latent_obs_dim, action_dim):
        super(ActionDecoder, self).__init__()
        self.invdyn = build_mlp(
            input_dim= obs_dim + latent_obs_dim,
            hidden_dims=[512, 512, 512],
            output_dim=action_dim,
            activation=nn.ReLU,
            output_activation=None,
            use_batchnorm=False
        )
    
    def forward(self, obs, latent_obs):
        action = self.invdyn(torch.cat([obs, latent_obs], dim=-1))

        return action