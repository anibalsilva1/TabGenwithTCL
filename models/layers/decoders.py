from typing import Union, Callable
import flax.linen as nn
import jax.numpy as jnp

class Decoder(nn.Module):
    hidden_dim: Union[int, list]
    output_dim: int
    activation_fn: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        if isinstance(self.hidden_dim, (list, tuple)):
            for hidden_dim in self.hidden_dim[::-1]:
                x = nn.Dense(hidden_dim)(x)
                x = self.activation_fn(x)
                
        else:
            x = nn.Dense(self.hidden_dim)(x)
            x = self.activation_fn(x)
            
        x = nn.Dense(self.output_dim, name='output')(x)

        return x

class DGDecoder(nn.Module):
    hidden_dim: Union[int, list]
    output_dim: int
    embed_dim: int
    activation_fn: Callable = nn.silu
 
    @nn.compact
    def __call__(self, x):
        if isinstance(self.hidden_dim, (tuple, list)):
            for hidden_dim in self.hidden_dim[::-1]:
                x = nn.DenseGeneral(features=(hidden_dim, self.embed_dim), axis=(-2, -1))(x)
                x = self.activation_fn(x)
        
        else:
            x = nn.DenseGeneral(features=(self.hidden_dim, self.embed_dim), axis=(-2, -1))(x)
            x = self.activation_fn(x)
        
        x = nn.DenseGeneral(features=(self.output_dim, self.embed_dim), axis=(-2, -1))(x)
        return x