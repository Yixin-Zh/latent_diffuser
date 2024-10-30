import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def variance_loss(latent, threshold=1.0):
    std = torch.sqrt(latent.var(dim=0) + 1e-4)
    loss = F.relu(threshold - std).mean()
    return loss

class Agent:

    def __init__(self,
                visionencoder,
                obsencoder,
                obsdecoder,
                actionencoder,
                actiondecoder,
                device= "cpu",
                optim_params=None,):
        
        if optim_params is None:
            optim_params = {"lr": 1e-4, "weight_decay": 1e-6}
        
        self.model = nn.ModuleDict({
            "visionencoder": visionencoder,
            "obsencoder": obsencoder,
            "obsdecoder": obsdecoder,
            "actionencoder": actionencoder,
            "actiondecoder": actiondecoder,
        }).to(device)

        self.optimizer = optim.AdamW(self.model.parameters(), **optim_params)

    def loss(self, next_image, next_image_pred, action, action_pred):
        action_loss = F.mse_loss(action_pred, action)
        image_loss = F.mse_loss(next_image_pred, next_image)
        return action_loss, image_loss
        

    def update(self,image, action, step):
        # import time
        # t1 = time.time()
        B, T, C, H, W = image.shape
        next_image = image[:, 1:].reshape(-1, C, H, W)
        
        image = image.reshape(B*T, C, H, W)
    
        emb = self.model["visionencoder"](image)
        emb = emb.reshape(B, T, -1)

        cur_obs = emb[:, :-1].reshape(-1, *emb.shape[2:])
        next_obs = emb[:, 1:].reshape(-1, *emb.shape[2:])
        action = action[:, :-1].reshape(-1, *action.shape[2:])

        
        latent_act = self.model["actionencoder"](cur_obs, next_obs) #(B, t, dim)
        latent_obs = self.model["obsencoder"](cur_obs, action)

        next_image_pred = self.model["obsdecoder"](cur_obs, latent_act)
        action_pred = self.model["actiondecoder"](cur_obs, latent_obs)
        

        action_loss, image_loss = self.loss(
            next_image, next_image_pred, action, action_pred
        )

        var_emb_loss = variance_loss(emb)
        var_obs_loss = variance_loss(latent_obs)
        var_act_loss = variance_loss(latent_act)    
       
           
        var_loss = 0.04*(var_emb_loss + var_obs_loss + var_act_loss)
        self.optimizer.zero_grad()
        
        (action_loss + image_loss + var_loss).backward()
        self.optimizer.step()
        # print("Time taken for one step: ", time.time()-t1)
        # print("Step: ", step)
        if step % 200 == 0:
            log = {
            "step": step,
            "action_loss": action_loss.item(),
            "image_loss": image_loss.item(),
            "var_loss": var_loss.item(),
            }
            return log
        else:
            return {}
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
        
    def eval(self):
        self.model.eval()