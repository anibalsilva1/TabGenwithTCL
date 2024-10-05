import jax.numpy as jnp
import flax.linen as nn

class Tokenizer(nn.Module):
    variable_indices: dict
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        xs = []
        for var, idxs in self.variable_indices.items():
            x_ = x[:, idxs]
            emb = nn.Dense(features=self.embed_dim, 
                           name=var, 
                           kernel_init=nn.initializers.kaiming_uniform(),
                           bias_init=nn.initializers.zeros
                           )(x_)
            xs.append(emb)
        
        return jnp.stack(xs, axis=1)
    
class Detokenizer(nn.Module):
    variable_indices: dict
    
    @nn.compact
    def __call__(self, x):
        x_out = []
        for i, (var, idxs) in enumerate(self.variable_indices.items()):
            x_ = x[:, i]
            dim = jnp.size(idxs)
            e = nn.Dense(features=dim, 
                         name=var,
                         kernel_init=nn.initializers.xavier_uniform(),
                         bias_init=nn.initializers.zeros
                         )(x_)
            x_out.append(e)
        
        x_out = jnp.concatenate(x_out, axis=1)
        return x_out