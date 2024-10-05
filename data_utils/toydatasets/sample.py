import pandas as pd
import jax
from jax import random
import jax.numpy as jnp

from typing import Any, Dict

Trainer = Any
Loader = Any
def sample_syn_from_model(
        trainer: Trainer,
        loader: Loader,
        n_samples: int,
        seed: int,
        variable_indices: Dict,
        transformations: Dict,
        ):
    
    main_rng = random.key(seed)
    n_classes = loader.dataset.n_classes
    ys = loader.dataset.label
    
    z_rng, eps_rng, y_rng = random.split(main_rng, num=3)
    ys = loader.dataset.label
    y_ = jnp.argmax(ys, axis=-1)
    y_sampled = random.choice(key=y_rng, a=y_, shape=(n_samples, ))
    y_sampled_oh = jax.nn.one_hot(y_sampled, num_classes=n_classes)
    
    if trainer.model_class.__name__ == "VAE":
        z = random.normal(key=z_rng, shape=(n_samples, trainer.model.latent_dim))
    elif trainer.model_class.__name__ == "Transformed":
        z = random.normal(key=z_rng, shape=(n_samples, len(variable_indices)+1, trainer.model.embed_dim))
    else:
        z = random.normal(key=z_rng, shape=(n_samples, trainer.model.latent_dim, trainer.model.embed_dim))
    
    x_recon = trainer.state.apply_fn(trainer.state.params, 
                                     rngs={'eps': eps_rng}, 
                                     z=z, 
                                     c=y_sampled_oh if trainer.model.conditional else None, 
                                     method="sample_from_latent")

    
    num_transf = transformations['quantile']
    x_syn = num_transf.inverse_transform(x_recon)
    df_syn = pd.DataFrame(x_syn, columns=list(variable_indices.keys()))
    df_syn['class'] = y_sampled
    return df_syn