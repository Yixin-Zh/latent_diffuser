import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import build_mlp
from .encoder import ResNet18Enc
from .decoder import ResNet18Dec


class ActionCVAE(nn.Module):
    """
    Given obs and latent_obs as condition, predict action
    """
    def __init__(self, obs_dim, latent_obs_dim, action_dim):

        super(ActionCVAE, self).__init__()
        z_dim = 16
        self.z_dim = z_dim
        self.encoder = build_mlp(
            input_dim= obs_dim + latent_obs_dim + action_dim,
            hidden_dims=[512, 512, 256],
            output_dim=z_dim*2,
            activation=nn.ReLU,
            output_activation=None,
            use_batchnorm=True)
        
        self.decoder = build_mlp(
            input_dim= obs_dim + latent_obs_dim + z_dim,
            hidden_dims=[512, 512, 256],
            output_dim=action_dim,
            activation=nn.ReLU,
            output_activation=None,
            use_batchnorm=True)
    
    def encode(self, obs, latent_obs, action):
        
        condition = torch.cat([obs, latent_obs,], dim=-1)

        mu, logvar = torch.chunk(self.encoder(torch.cat([condition, action], dim=-1)), 2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, obs, latent_obs, z):

        condition = torch.cat([obs, latent_obs,], dim=-1)
       
        return self.decoder(torch.cat([condition, z], dim=-1))
    
    def forward(self, obs, latent_obs, action):
        mu, logvar = self.encode(obs, latent_obs, action)
        z = self.reparameterize(mu, logvar)
        return [self.decode(obs, latent_obs, z), mu, logvar]
    
    def inference(self, obs, latent_obs):
        z = torch.randn(obs.shape[0], self.z_dim).to(obs.device)
        return self.decode(obs, latent_obs, z)
    
class ObsCVAE(nn.Module):
    """
    Given obs and latent_act as condition, predict next_obs
    """

    def __init__(self, obs_dim, latent_act_dim):
        super(ObsCVAE, self).__init__()
        condition_dim = obs_dim + latent_act_dim

        self.z_dim = 128
        self.encoder = ResNet18Enc(z_dim=2*self.z_dim, nc=4) # input for image and condition
        self.decoder = ResNet18Dec(obs_dim=condition_dim,
                                      latent_act_dim=self.z_dim,
                                      nc=3) # output for image
        
        self.condition_emb = build_mlp(
            input_dim=condition_dim,
            hidden_dims=[256,],
            output_dim=84*84,
            activation=nn.ReLU,
            output_activation=None,
            use_batchnorm=True)
        
    def encode(self, image, obs, latent_act):
        
        condition = self.condition_emb(torch.cat([obs, latent_act], dim=-1))
        condition = condition.view(-1, 1, 84, 84)

        mu, logvar = torch.chunk(self.encoder(torch.cat([image, condition], dim=1)), 2, dim=-1)
        return mu, logvar
    
    def repamariterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, obs, latent_act, z):
        condition = torch.cat([obs, latent_act], dim=-1)
        return self.decoder(condition, z)
    
    def forward(self, image, obs, latent_act):
        mu, logvar = self.encode(image, obs, latent_act)
        z = self.repamariterize(mu, logvar)
        return [self.decode(obs, latent_act, z), mu, logvar]
    
    def inference(self, obs, latent_act):
        z = torch.randn(obs.shape[0], self.z_dim).to(obs.device)
        return self.decode(obs, latent_act, z)
    

        