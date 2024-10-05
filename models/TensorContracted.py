import flax.linen as nn
import jax.numpy as jnp
from jax import random

from .layers import DGEncoder, DGDecoder, Tokenizer, Detokenizer

class TensorContracted(nn.Module):
    variable_indices: dict
    output_dim: int
    hidden_dim: int = 96
    latent_dim: int = 32
    embed_dim: int = 4
    conditional: bool = True

    def setup(self):
        self.eps_rng = self.make_rng('eps')

        self.tokenizer = Tokenizer(variable_indices=self.variable_indices, embed_dim=self.embed_dim)
        
        if self.conditional:
            self.conditional_tokenizer = nn.Dense(features=self.embed_dim,
                                                  kernel_init=nn.initializers.kaiming_uniform(),
                                                  bias_init=nn.initializers.zeros,
                                                  name="conditional_tokenizer")
    

        self.encoder = DGEncoder(hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, embed_dim=self.embed_dim)
        self.decoder = DGDecoder(hidden_dim=self.hidden_dim, output_dim=self.output_dim, embed_dim=self.embed_dim)
        
        self.detokenizer = Detokenizer(variable_indices=self.variable_indices)

    def __call__(self, x, c=None, training=True):
        x = self.tokenizer(x)
        if self.conditional:
            c = self.conditional_tokenizer(c)
            c = c[:, None, :]
            x = jnp.concatenate([x, c], axis=1)

        mu, logvar = self.encoder(x)

        z = self.reparametrize(mu, logvar)
        if self.conditional:
            z = jnp.concatenate([z, c], axis=1)

        x_recon = self.decoder(z)
        x_recon = self.detokenizer(x_recon)
        return x_recon, mu, logvar, z

    
    def reparametrize(self, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(self.eps_rng, shape=std.shape)
        z = mu + std * eps
        return z
    
    def sample_from_latent(self, z, c=None):
        if self.conditional:
            c = self.conditional_tokenizer(c)
            c = c[:, None, :]
            z = jnp.concatenate([z, c], axis=1)

        x_recon = self.decoder(z)
        x_recon = self.detokenizer(x_recon)
        
        return x_recon