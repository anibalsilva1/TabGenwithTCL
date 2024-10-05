import jax.numpy as jnp
from jax import random
import jax

from data_utils import process_output
from . import *

def sample_from_model(
        trainer,
        loader,
        n_samples,
        seed,
        meta,
        transformations,
        ):
    
    target_name = meta['target_name']
    
    main_rng = random.key(seed)
    n_classes = loader.dataset.n_classes
    

    z_rng, eps_rng, y_rng = random.split(main_rng, num=3)
    if trainer.model.conditional:
        ys = loader.dataset.label
        y_ = jnp.argmax(ys, axis=-1)
        y_sampled = random.choice(key=y_rng, a=y_, shape=(n_samples, ))
        y_sampled_oh = jax.nn.one_hot(y_sampled, num_classes=n_classes)
    
    if trainer.model_class.__name__ == "VAE":
        z = random.normal(key=z_rng, shape=(n_samples, trainer.model.latent_dim))
    elif trainer.model_class.__name__ == "Transformed":
        z = random.normal(key=z_rng, shape=(n_samples, len(meta['variable_indices'])+1, trainer.model.embed_dim))
    else:
        z = random.normal(key=z_rng, shape=(n_samples, trainer.model.latent_dim, trainer.model.embed_dim))
    
    x_recon = trainer.state.apply_fn(trainer.state.params, 
                                     rngs={'eps': eps_rng}, 
                                     z=z, 
                                     c=y_sampled_oh if trainer.model.conditional else None, 
                                     method="sample_from_latent")

    df_syn = process_output(x_recon, meta, transformations)
    if trainer.model.conditional:
        df_syn[target_name] = y_sampled
    
    return df_syn

def get_model(model_name):
    if model_name == "VAE":
        model_class = VAE
    elif model_name == "TensorContracted":
        model_class = TensorContracted
    elif model_name == "TensorConFormer":
        model_class = TensorConFormer
    elif model_name == "Transformed":
        model_class = Transformed
    elif model_name == "TensorConFormerEnc":
        model_class = TensorConFormerEnc
    elif model_name == "TensorConFormerDec":
        model_class = TensorConFormerDec
    else:
        raise ValueError(f"Model: {model_name} not found.")
    return model_class


def get_embeddings(trainer, x, c, iters, meta):
    embeddings_x = {}
    embeddings_z = {}
    embeddings_y = {}
    weights = {}
    
    n_features = len(meta['variable_indices'])
    for iter in iters:
        state, *metrics = trainer.load_from_checkpoint(step=iter)
        outs, inter = state.apply_fn(variables=state.params, rngs={'eps': random.key(0)}, x=x, c=c, capture_intermediates=True)
        x_recon, mu, logvar, z = outs
        y_embeddings = inter['intermediates']['conditional_tokenizer']['__call__'][0]
        if trainer.model_class.__name__ == "TensorContracted":
            x_embeddings = inter['intermediates']['decoder']['__call__'][0]
        else:
            x_embeddings = inter['intermediates']['decoder_transformer']['__call__'][0]
        
        embeddings_x[iter] = x_embeddings
        embeddings_z[iter] = z[:, :-1, :]
        embeddings_y[iter] = y_embeddings
        weights[iter] = state.params['params']
        
        
    return weights, embeddings_x, embeddings_z, embeddings_y

def estimate_batchsize(n_val: int):
    if 0 < n_val <= 128:
        batch_size = 32
    elif 128 < n_val <= 512:
        batch_size = 64
    elif 512 < n_val <= 1024:
        batch_size = 128
    elif 1024 < n_val <= 2048:
        batch_size = 256
    elif 2048 < n_val < 4096:
        batch_size = 512
    else:
        batch_size = 1024
    
    return batch_size