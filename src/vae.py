from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_channels: int = 1, img_size: int = 28, latent_dim: int = 16, hidden_dims: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.img_size = img_size
        self.input_dim = input_channels * img_size * img_size
        self.latent_dim = latent_dim

        enc_layers = []
        prev = self.input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers = []
        rev_h = list(hidden_dims)[::-1]
        prev = latent_dim
        for h in rev_h:
            dec_layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        dec_layers += [nn.Linear(prev, self.input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    # Reparametrization trick (z = mu + std * eps)
    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        x_hat = x_hat.view(x_hat.size(0), 1, self.img_size, self.img_size)
        return x_hat
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar