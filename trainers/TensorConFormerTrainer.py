import jax
from jax import random

from .base import TrainerModule
from .loss import transformervae_loss
from models import TensorConFormer

class TensorConFormerTrainer(TrainerModule):

    def __init__(self,
                 indexes: dict = None,
                 **kwargs,
                 ):
        super().__init__(
            model_class=TensorConFormer,
            **kwargs)
        
        self.indexes = indexes
        
    def create_functions(self):

        def loss_fn(params, batch, batch_rng, training):

            batch_rng, eps_rng, drpt_rng = random.split(batch_rng, num=3)
            x, labels = batch
            x_recon, mu, logvar, _ = self.state.apply_fn(
                params,
                x=x,
                c=labels if self.model.conditional else None,
                rngs={'eps': eps_rng, 'dropout': drpt_rng},
                training=training,
            )
            
            metrics = transformervae_loss(x_recon, mu, logvar, x, self.indexes)
            loss = metrics['loss']
            
            return loss, metrics

        def train_step(state, batch, batch_rng):

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            res, grads = grad_fn(state.params, batch, batch_rng, training=True)
            
            _, metrics = res[0], res[1]
            state = state.apply_gradients(grads=grads)
            return state, metrics

        def eval_step(state, batch, batch_rng):
            
            res = loss_fn(state.params, batch, batch_rng, training=False)
            _, metrics = res[0], res[1]
            return metrics
        
        return train_step, eval_step