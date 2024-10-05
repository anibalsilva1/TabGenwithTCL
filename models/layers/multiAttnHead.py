import flax.linen as nn
import jax.numpy as jnp


from .utils import scaled_dot_product


class MultiheadAttention(nn.Module):
    embed_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads (h)

    def setup(self):
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        
        if self.num_heads > 1:
            self.o_proj = nn.Dense(self.embed_dim,
                                   kernel_init=nn.initializers.xavier_uniform(),
                                   bias_init=nn.initializers.zeros)

    def __call__(self, x):
        batch_size, seq_length, embed_dim = x.shape
        
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        out = values.reshape(batch_size, seq_length, embed_dim)
        
        if self.num_heads > 1:
            out = self.o_proj(out)

        return out, attention