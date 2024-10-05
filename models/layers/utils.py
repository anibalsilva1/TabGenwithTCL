import flax.linen as nn
import jax.numpy as jnp


def scaled_dot_product(q, k, v):
    d_k = q.shape[-1]
    attn_logits = jnp.einsum('...ij, ...kj -> ...ik', q, k)
    attn_logits = attn_logits / jnp.sqrt(d_k)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.einsum('...ij, ...jk -> ...ik', attention, v)
    return values, attention