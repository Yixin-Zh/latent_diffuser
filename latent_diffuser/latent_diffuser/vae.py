import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_diffuser.nn import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self, device='cpu'):
        super(VAE, self).__init__()
        self.encoder = Encoder(ch=64, out_ch=64, ch_mult=(1,2,4,8,16), num_res_blocks=5,
                      attn_resolutions=[4], dropout=0.0, resamp_with_conv=True,
                      in_channels=3, resolution=84, z_channels=4, double_z=True)
        
        self.decoder = Decoder(ch=64, out_ch=3, ch_mult=(1,2,4,8,16), num_res_blocks=5,
                        attn_resolutions=[4], dropout=0.0, resamp_with_conv=True,
                        in_channels=4, resolution=5, z_channels=4, give_pre_end=False, tanh_out=False)
        self.models = nn.ModuleDict({
            "encoder": self.encoder,
            "decoder": self.decoder,
        }).to(device)

      
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=1e-4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar).to(mu.device)
        eps = torch.randn_like(std).to(mu.device)
        return mu + eps*std
    
    def forward(self, x):
        z_ = self.encoder(x)
        B, C, H, W = z_.shape
        mu, logvar = z_[:, :C//2], z_[:, C//2:]
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def update(self, x, step):
        x_recon, mu, logvar = self.forward(x)
        rec_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = rec_loss + 5e-6 * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % 200 == 0:
            return {
                "step": step,
                "loss": loss.item(),
                "rec_loss": rec_loss.item(),
                "kl_loss": kl_loss.item(),
            }
        else:
            return {}

    def sample_latent(self, x):
        with torch.no_grad():
            z_ = self.encoder(x)
            B, C, H, W = z_.shape
            mu, logvar = z_[:, :C//2], z_[:, C//2:]
            # z = self.reparameterize(mu, logvar)
            return mu #s shape: (B, C//2, H, W)
        
    
    def save(self, path):
        torch.save(self.models.state_dict(), path)
    
    def load(self, path):
        self.models.load_state_dict(torch.load(path))
    
    def eval(self):
        return super().eval()