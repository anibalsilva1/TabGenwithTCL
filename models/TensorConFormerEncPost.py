import jax.numpy as jnp
from jax import random
import flax.linen as nn

from .layers import Tokenizer, Detokenizer, TensoFormerEncoder, DGDecoder, DGEncoder
from models import TransformerEncoder


class TensorConFormerEncPost(nn.Module):
    variable_indices: dict
    output_dim: int
    num_layers: int = 2
    embed_dim: int = 4
    conditional_embed_dim: int = 4
    latent_dim: int = 32
    num_heads: int = 1
    dim_feedforward: int = 128
    dropout_prob: float = 0.0
    hidden_dim: int = 96
    conditional: bool = True

    def setup(self):
        self.eps_rng = self.make_rng("eps")

        self.tokenizer = Tokenizer(variable_indices=self.variable_indices,
                                   embed_dim=self.embed_dim)
        
        if self.conditional:
            self.conditional_tokenizer = nn.Dense(features=self.conditional_embed_dim, 
                                                  kernel_init=nn.initializers.kaiming_uniform(),
                                                  bias_init=nn.initializers.zeros,
                                                  name="conditional_tokenizer")
        
        self.encoder_transformer = TransformerEncoder(num_layers=self.num_layers,
                                                      embed_dim=self.embed_dim,
                                                      num_heads=self.num_heads,
                                                      dim_feedforward=self.dim_feedforward,
                                                      dropout_prob=self.dropout_prob)

        self.encoder = TensoFormerEncoder(hidden_dim=self.hidden_dim,
                                          latent_dim=self.latent_dim,
                                          embed_dim=self.embed_dim)
        
        self.mu = TransformerEncoder(num_layers=self.num_layers,
                                     embed_dim=self.embed_dim,
                                     num_heads=self.num_heads,
                                     dim_feedforward=self.dim_feedforward,
                                     dropout_prob=self.dropout_prob)

        self.logvar = TransformerEncoder(num_layers=self.num_layers,
                                         embed_dim=self.embed_dim,
                                         num_heads=self.num_heads,
                                         dim_feedforward=self.dim_feedforward,
                                         dropout_prob=self.dropout_prob)

        self.decoder = DGDecoder(hidden_dim=self.hidden_dim,
                                    embed_dim=self.embed_dim,
                                    output_dim=self.output_dim)
        
        
        self.detokenizer = Detokenizer(variable_indices=self.variable_indices)
        
    def __call__(self, x, c=None, training=True):
        x = self.tokenizer(x)
        if self.conditional:
            c = self.conditional_tokenizer(c)
            c = c[:, None, :]
            x = jnp.concatenate([x, c], axis=1)
        
        x = self.encoder_transformer(x, training=training)
        x = self.encoder(x)

        mu = self.mu(x, training=training)
        logvar = self.logvar(x, training=training)
        
        z = self.reparametrize(mu, logvar)
        if self.conditional:
            z = jnp.concatenate([z, c], axis=1)

        x_ = self.decoder(z)
        x_recon = self.detokenizer(x_)
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

        x_ = self.decoder(z)
        x_recon = self.detokenizer(x_)
        
        return x_recon
