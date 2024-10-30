import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def variance_loss(latent, threshold=1.0):
    std = torch.sqrt(latent.var(dim=0) + 1e-4)
    loss = F.relu(threshold - std).mean()
    return loss


class AgentVAE:
    def __init__(self, 
                 visionencoder,
                 obsencoder,
                 actionencoder,
                 obsdecoder,
                 actiondecoder,
                 device= "cpu",
                optim_params=None,):
        
        if optim_params is None:
            optim_params = {"lr": 1e-4, "weight_decay": 1e-6}
        
        self.model = nn.ModuleDict({
            "visionencoder": visionencoder,
            "obsencoder": obsencoder,
            "actionencoder": actionencoder,
            "obsdecoder": obsdecoder,
            "actiondecoder": actiondecoder,
        }).to(device)

        self.optimizer = optim.AdamW(self.model.parameters(), **optim_params)
    
    def loss(self, next_image, next_image_pred, action, action_pred):

        predicted_image, mu1, logvar1 = next_image_pred[:3]

        img_rec_loss = F.mse_loss(predicted_image, next_image)
        img_kl_loss = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        predited_action, mu2, logvar2 = action_pred[:3]

        act_rec_loss = F.mse_loss(predited_action, action)
        act_kl_loss = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())

        return img_rec_loss, 0.00025*img_kl_loss, act_rec_loss, 0.00025*act_kl_loss
    

    def update(self, image, action, step):
        """
        image: (B, T, C, H, W)
        action: (B, T, dim)
        """
        
        B, T, C, H, W = image.shape
        next_image = image[:, 1:].reshape(-1, C, H, W)
        
        image = image.reshape(B*T, C, H, W)
    
        emb = self.model["visionencoder"](image)
        emb = emb.reshape(B, T, -1)

        cur_obs = emb[:, :-1].reshape(-1, *emb.shape[2:])
        next_obs = emb[:, 1:].reshape(-1, *emb.shape[2:])
        action = action[:, :-1].reshape(-1, *action.shape[2:])

        
        latent_act = self.model["actionencoder"](cur_obs, next_obs)
        latent_obs = self.model["obsencoder"](cur_obs, action)

        next_image_pred_ = self.model["obsdecoder"](next_image, cur_obs, latent_act)
        
        action_pred_ = self.model["actiondecoder"](cur_obs, latent_obs, action)

        image_loss, img_kl_loss, action_loss, act_kl_loss = self.loss(
            next_image, next_image_pred_, action, action_pred_
        )

        var_emb_loss = variance_loss(emb)
        var_obs_loss = variance_loss(latent_obs)
        var_act_loss = variance_loss(latent_act)

        var_loss = (var_emb_loss + var_obs_loss + var_act_loss)*0.04

        self.optimizer.zero_grad()
        (image_loss+img_kl_loss+action_loss+act_kl_loss+var_loss).backward()
        self.optimizer.step()
      
        if step % 200 == 0: 
            log = {
            "step": step,
            "image_loss": image_loss.item(),
            "img_kl_loss": img_kl_loss.item(),
            "action_loss": action_loss.item(),
            "act_kl_loss": act_kl_loss.item(),
            "var_loss": var_loss.item(),
            }
            return log
        else:
            return {}

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def eval(self):
        self.model.eval()
