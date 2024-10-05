from jax import random
import jax

from .base import TrainerModule
from .loss import vae_loss

from models import VAE

from typing import Callable

class VAETrainer(TrainerModule):

    def __init__(self,
                 l2_regularizer: Callable = False,
                 indexes: dict = None,
                 **kwargs,
                 ):
        super().__init__(
            model_class=VAE,
            **kwargs)
        
        self.indexes = indexes
        self.l2_regularizer = l2_regularizer
    
    def create_functions(self):

        def loss_fn(params, batch, batch_rng):

            batch_rng, eps_rng, drpt_rng = random.split(batch_rng, num=3)
            x, conditionals = batch
            x_recon, mu, logvar, _ = self.state.apply_fn(
                params, 
                x=x,
                c=conditionals if self.model.conditional else None,
                rngs={'eps': eps_rng, 'dropout': drpt_rng},
            )
            
            metrics = vae_loss(x_recon, mu, logvar, x, self.indexes)
            loss = metrics['loss']
            if self.l2_regularizer:
                loss += self.l2_regularizer(params)
            
            return loss, metrics

        def train_step(state, batch, batch_rng):

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            res, grads = grad_fn(state.params, batch, batch_rng)
            
            _, metrics = res[0], res[1]
            state = state.apply_gradients(grads=grads)
            return state, metrics

        def eval_step(state, batch, batch_rng):
            
            res = loss_fn(state.params, batch, batch_rng)
            _, metrics = res[0], res[1]
            return metrics
        
        return train_step, eval_step