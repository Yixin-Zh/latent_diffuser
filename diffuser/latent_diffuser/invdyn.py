import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser.nn_diffusion import DiT1d
from diffuser.diffusion import ContinuousDiffusionSDE
from diffuser.utils import report_parameters


class InvPolicy:
    '''
    Inverse Dynamics Model
    condition on latent_obs(z_t and z_{t+1})
    predict action (a_t)
    Just one step prediction
    '''
    def __init__(self, device):
        self.model = DiT1d(
        7, 4*5*5*2,
        d_model=600, n_heads=10, depth=4, timestep_emb_type="fourier")
        print(f"======================= Parameter Report of Inverse Dynamics Model =======================")
        report_parameters(self.model)
        print(f"==============================================================================")

        # ----------------- Masking -------------------
        # only one step prediction
        fix_mask = torch.zeros((1, 7)) 
        loss_weight = torch.ones((1, 7)) 

        self.diffuser = ContinuousDiffusionSDE(
        self.model,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.995,
        device=device, predict_noise=True, noise_schedule="linear")
    
    def train(self):
        self.diffuser.train()

    def update(self, latent_obs, act, step):
        assert len(latent_obs.shape) == 3
        assert len(act.shape) == 3
        return self.diffuser.update(x0= act, condition=latent_obs, step=step)   

    def save(self, path):
        self.diffuser.save(path)
    
    def load(self, path):
        self.diffuser.load(path)
    
    def eval(self):
        self.diffuser.eval()    
    
    