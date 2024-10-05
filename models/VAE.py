from typing import Union

import jax.numpy as jnp
from jax import random
import flax.linen as nn

from .layers import Encoder, Decoder

class VAE(nn.Module):
    output_dim: int
    hidden_dim: Union[int, list] = 512
    latent_dim: int = 256
    conditional: bool = True

    def setup(self):
        self.eps_rng = self.make_rng('eps')

        self.encoder = Encoder(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.hidden_dim, self.output_dim)

    def __call__(self, x, c=None):
        if self.conditional:
            x = jnp.concatenate([x, c], axis=-1)

        mu, logvar = self.encoder(x)

        z = self.reparametrize(mu, logvar)
        
        if self.conditional:
            z = jnp.concatenate([z, c], axis=-1)
            
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z
    
    def reparametrize(self, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(key=self.eps_rng, shape=std.shape)
        return mu + eps * std
    
    def sample_from_latent(self, z, c=None):
        if self.conditional:
            z = jnp.concatenate([z, c], axis=-1)
        return self.decoder(z)
    