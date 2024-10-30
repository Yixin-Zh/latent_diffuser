import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffuser.bc.functions as functions

from diffuser.nn.encoder import ResNet18Enc 
from diffuser.nn.decoder import ResNet18DecV2
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()

        self.trunk = functions.build_mlp(state_dim, action_dim, 
            n_layers, hidden_dim, activation='relu', output_activation='tanh')
        self.apply(functions.weight_init)

    def forward(self, state):
        h = self.trunk(state)
        return h
class BCObsActAgent:
    def __init__(self, obs_dims, act_dims, device, n_layers=4, hidden_dim=512, lr=3e-4):

        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        self.act_dim = act_dims['act_dim']
        self.lat_act_dim = act_dims['lat_act_dim']
        self.device = device
        self.batch_norm = False


        """
        1. obs_enc: obs -> lat_obs;
        2. obs_dec: lat_obs -> obs;

        3. act_enc: lat_obs + act(no-gripper) -> lat_act;
        4. act_dec: lat_obs + lat_act -> act(no-gripper);

        5. inv_dyn: lat_obs + lat_next_obs -> lat_act;
        6. fwd_dyn: lat_obs + lat_act -> lat_next_obs;
        
        7. actor: lat_obs -> lat_act ***one more dim for gripper action***
        
        """
        self.obs_enc = ResNet18Enc(z_dim=self.lat_obs_dim).to(device)
        self.obs_enc_opt = torch.optim.Adam(self.obs_enc.parameters(), lr=lr)

        self.obs_dec = ResNet18DecV2(z_dim=self.lat_obs_dim).to(device)
        self.obs_dec_opt = torch.optim.Adam(self.obs_dec.parameters(), lr=lr)
        
        self.act_enc = functions.build_mlp(self.lat_obs_dim+self.act_dim-1, self.lat_act_dim, n_layers, 
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.act_enc_opt= torch.optim.Adam(self.act_enc.parameters(), lr=lr)

        self.act_dec = functions.build_mlp(self.lat_obs_dim+self.lat_act_dim, self.act_dim-1, n_layers,
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.act_dec_opt = torch.optim.Adam(self.act_dec.parameters(), lr=lr)

        self.inv_dyn = functions.build_mlp(self.lat_obs_dim*2, self.lat_act_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=lr)

        self.fwd_dyn = functions.build_mlp(self.lat_obs_dim+self.lat_act_dim, self.lat_obs_dim, 
            n_layers, hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.fwd_dyn_opt = torch.optim.Adam(self.fwd_dyn.parameters(), lr=lr)

        # One more dim for gripper action
        self.actor = Actor(self.lat_obs_dim, self.lat_act_dim+1, n_layers, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.modules = [self.obs_enc, self.obs_dec, self.act_enc, 
            self.act_dec, self.inv_dyn, self.fwd_dyn, self.actor]
    
    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
    
            lat_obs = self.obs_enc(obs)
            gen_img = self.obs_dec(lat_obs)[0]
            
            lat_act = self.actor(lat_obs)
            lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
            act = self.act_dec(torch.cat([lat_obs, lat_act], dim=-1))
            act = torch.cat([act, gripper_act], dim=-1)
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        
        return act, (gen_img.cpu().data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    def test_latent(self, obs, act, deterministic=False):
        with torch.no_grad():
            
            act = act[:,:-1]
            B, T, C, H, W = obs.shape
            # emb: (B, T, lat_obs_dim)
            emb = self.obs_enc(obs.reshape(-1, C, H, W)).reshape(B, T, -1)

            # lat_obs: (B, T-1, lat_obs_dim) -> (B*(T-1), lat_obs_dim)
            lat_obs = emb[:, :-1].reshape(-1, self.lat_obs_dim)
            lat_act = self.act_enc(torch.cat([lat_obs, act], dim=-1))

        return lat_obs, lat_act


    def update_actor(self, obs, act, step):
        """
        args:
        obs: (B, T, C, H, W)
        act: (B, T-1, dim)
        All functions:
        1. obs_enc: robot_obs -> lat_obs;
        2. actor: lat_obs + obj_obs -> lat_act;
        3. lat_act -> lat_act, gripper_act;
        4. act_dec: robot_obs + lat_act -> act;

        actor: lat_obs-> lat_act
        """
       
        # obs_enc: robot_obs -> lat_obs
        B, T, C, H, W = obs.shape
        lat_obs = self.obs_enc(obs[:, :-1].reshape(-1, C, H, W))
        # actor: lat_obs + obj_obs -> lat_act
        lat_act = self.actor(lat_obs)
        # seperate: lat_act -> lat_act, gripper_act
        lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
        # act_dec: robot_obs + lat_act -> act
        pred_act = self.act_dec(torch.cat([lat_obs, lat_act], dim=-1))
        
        pred_act = torch.cat([pred_act, gripper_act], dim=-1)
        loss = F.mse_loss(pred_act, act)

        self.actor_opt.zero_grad()
        self.obs_enc_opt.zero_grad()
        self.act_dec_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.obs_enc_opt.step()
        self.act_dec_opt.step()
        
        if step % 200 == 0:
            return {"step": step, "actor_loss": loss.item()}
        else:
            return {}
        

    def update_dyn_cons(self, obs, act, step):
        """
        Args:
        obs: (B, T, C, H, W)
        act: (B, T-1, dim)
        Parameters:
        For obs, only encode robot_obs;
        For act, only encode ee_act;  
        obs: img
        act: ee_act + gripper_act
        All functions:
        1. obs_enc: robot_obs -> lat_obs;
        2. obs_dec: lat_obs -> robot_obs;

        3. act_enc: lat_obs + act -> lat_act;
        4. act_dec: lat_obs + lat_act -> act;

        5. inv_dyn: lat_obs + lat_next_obs -> lat_act;
        6. fwd_dyn: lat_obs + lat_act -> lat_next_obs;
        """
        
        act = act[:,:-1]
        B, T, C, H, W = obs.shape
        # emb: (B, T, lat_obs_dim)
        emb = self.obs_enc(obs.reshape(-1, C, H, W)).reshape(B, T, -1)

        # lat_obs: (B, T-1, lat_obs_dim) -> (B*(T-1), lat_obs_dim)
        lat_obs = emb[:, :-1].reshape(-1, self.lat_obs_dim)
        # lat_next_obs: (B, T-1, lat_obs_dim) -> (B*(T-1), lat_obs_dim)
        lat_next_obs = emb[:, 1:].reshape(-1, self.lat_obs_dim)

        # inv_loss: in act space
        pred_lat_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        pred_act = self.act_dec(torch.cat([lat_obs, pred_lat_act], dim=-1))
        inv_loss = F.mse_loss(pred_act, act)

        # fwd_loss: in lat_obs space
        lat_act = self.act_enc(torch.cat([lat_obs, act], dim=-1))
        pred_next_obs = self.fwd_dyn(torch.cat([lat_obs, lat_act], dim=-1))
        fwd_loss = F.mse_loss(pred_next_obs, lat_next_obs)    

        # recon loss: in robot_obs space and act space
        recon_robot_obs = self.obs_dec(lat_obs)
        target_obs = obs[:, :-1].reshape(-1, C, H, W)
        recon_obs_loss = F.mse_loss(recon_robot_obs, target_obs)
        recon_act = self.act_dec(torch.cat([lat_obs, lat_act], dim=-1))
        recon_act_loss = F.mse_loss(recon_act, act)

        loss = fwd_loss + inv_loss + recon_obs_loss + recon_act_loss

        self.obs_enc_opt.zero_grad()
        self.obs_dec_opt.zero_grad()
        self.act_enc_opt.zero_grad()
        self.act_dec_opt.zero_grad()
        self.inv_dyn_opt.zero_grad()
        self.fwd_dyn_opt.zero_grad()
        loss.backward()
        self.obs_enc_opt.step()
        self.obs_dec_opt.step()
        self.act_enc_opt.step()
        self.act_dec_opt.step()
        self.inv_dyn_opt.step()    
        self.fwd_dyn_opt.step()

        if step % 200 == 0:
            return {"step": step, 
                    "dyn_loss": loss.item(),
                    "fwd_loss": fwd_loss.item(),
                    "inv_loss": inv_loss.item(),
                    "recon_obs_loss": recon_obs_loss.item(),
                    "recon_act_loss": recon_act_loss.item(),
                    "lat_obs_sq": (lat_obs**2).mean().item(),
                    "pred_lat_act_sq": (pred_lat_act**2).mean().item()}
        else:
            return {}

    def update(self, obs, act, step):
       
        log1 = self.update_dyn_cons(obs, act, step)
        log2 = self.update_actor(obs, act, step)
        return {**log1, **log2}

    def save(self, model_dir, name):
        
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc_{name}.pt')
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec_{name}.pt')
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc_{name}.pt')
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec_{name}.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn_{name}.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn_{name}.pt')
        torch.save(self.actor.state_dict(), f'{model_dir}/actor_{name}.pt')


    def load(self, model_dir, name):
        self.obs_enc.load_state_dict(torch.load(f'{model_dir}/obs_enc_{name}.pt'))
        self.obs_dec.load_state_dict(torch.load(f'{model_dir}/obs_dec_{name}.pt'))
        self.act_enc.load_state_dict(torch.load(f'{model_dir}/act_enc_{name}.pt'))
        self.act_dec.load_state_dict(torch.load(f'{model_dir}/act_dec_{name}.pt'))
        self.inv_dyn.load_state_dict(torch.load(f'{model_dir}/inv_dyn_{name}.pt'))
        self.fwd_dyn.load_state_dict(torch.load(f'{model_dir}/fwd_dyn_{name}.pt'))
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor_{name}.pt'))
    
    def eval(self):
        self.obs_enc.eval()
        self.obs_dec.eval()
        self.act_enc.eval()
        self.act_dec.eval()
        self.inv_dyn.eval()
        self.fwd_dyn.eval()
        self.actor.eval()
