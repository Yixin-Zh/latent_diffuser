import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser.nn_diffusion import DiT1d
from diffuser.diffusion import ContinuousDiffusionSDE
from diffuser.utils import report_parameters

class Planner:
    '''
    Latent Planner
    condition on latent_obs(z_t)
    predict latent_obs_seq(z_{t+1}, z_{t+2}, ..., z_{t+H})
    '''
    def __init__(self, horizon, device):
        self.latent_dim = 4*5*5
        self.model = DiT1d(
        self.latent_dim, self.latent_dim,
        d_model=700, n_heads=20, depth=5, timestep_emb_type="fourier")
        print(f"======================= Parameter Report of Inverse Dynamics Model =======================")
        report_parameters(self.model)
        print(f"==============================================================================")

        # ----------------- Masking -------------------
        # only one step prediction
        fix_mask = torch.zeros((horizon, self.latent_dim))
        loss_weight = torch.ones((horizon, self.latent_dim))
        loss_weight[0] = 5

        self.diffuser = ContinuousDiffusionSDE(
        self.model,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.995,
        device=device, predict_noise=True, noise_schedule="linear")
    
    def train(self):
        self.diffuser.train()

    def update(self, latent_obs, step):
        '''
        latent_obs: (B, T, C)
        use first latent as condition
        predict next latent sequence
        '''
        assert len(latent_obs.shape) == 3
        condition = latent_obs[:, 0].unsqueeze(1)
        seq = latent_obs[:, 1:]
        return self.diffuser.update(x0= seq, condition=condition, step=step)   

    def save(self, path):
        self.diffuser.save(path)
    
    def load(self, path):
        self.diffuser.load(path)
    
    def eval(self):
        self.diffuser.eval()    
    
    