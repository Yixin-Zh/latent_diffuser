import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_diffuser.nn_diffusion import DiT1d
from latent_diffuser.diffusion import ContinuousDiffusionSDE
from latent_diffuser.utils import report_parameters


class InvPolicy:
    '''
    Inverse Dynamics Model
    condition on latent_obs(z_t and z_{t+1})
    predict action (a_t)
    Just one step prediction
    '''
    def __init__(self, device = 'cpu'):
        self.latent_dim = 4*5*5
        self.model = DiT1d(
        7, self.latent_dim*2*2, # 2 for view; 2 for z_t and z_{t+1}
        d_model=700, n_heads=10, depth=4, timestep_emb_type="fourier")
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
        assert len(latent_obs.shape) == 2
        assert len(act.shape) == 3
        return self.diffuser.update(x0= act, condition=latent_obs, step=step)   

    def save(self, path):
        self.diffuser.save(path)
    
    def load(self, path):
        self.diffuser.load(path)
    
    def eval(self):
        self.diffuser.eval()    
    
    