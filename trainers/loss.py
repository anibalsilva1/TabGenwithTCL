import jax.numpy as jnp
import optax

def reconstruction_loss(x_recon, x, indices):
    reconstruction_loss = 0
    for col, idxs in indices.items():
        x_recon_, x_ = x_recon[:, idxs], x[:, idxs]
        if jnp.size(idxs) == 1:
            reconstruction_loss += optax.squared_error(x_recon_.squeeze(), x_.squeeze()).mean(0)
        elif jnp.size(idxs) > 1:
            reconstruction_loss += optax.softmax_cross_entropy(x_recon_, x_).mean(0)
        else:
            raise NotImplementedError(f"Variable type: {col} not expected.")
        
    return reconstruction_loss / len(indices)

def kl_div_tensor(mu, logvar):
    '''
    For VAETransformer, the latent space has dimensions [batch_dim, n_features, embed_dim].
    the KL Divergence is averaged over all dimensions, as in SynTab.
    '''
    temp = 1 + logvar - jnp.square(mu) - jnp.exp(logvar)
    kl_div = - 0.5 * jnp.mean(temp, axis=(-1, -2))
    return kl_div.mean()

def transformervae_loss(x_recon, mu, logvar, x, indices):
    rec_loss = reconstruction_loss(x_recon, x, indices)
    kl = kl_div_tensor(mu, logvar)
    return {'rec_loss': rec_loss, 'kl_div': kl, 'loss': rec_loss + kl}

def kl_div_vae(mu, logvar):
    return -0.5 * jnp.mean(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1).mean(0)

def vae_loss(x_recon, mu, logvar, x, indices=None):
    rec_loss = reconstruction_loss(x_recon, x, indices)
    kl = kl_div_vae(mu, logvar)
    return {'rec_loss': rec_loss, 'kl_div': kl, 'loss': rec_loss + kl}