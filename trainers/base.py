import flax.linen as nn
from jax import random
from flax.training import train_state
from flax.training.early_stopping import EarlyStopping

import pickle

import os
from copy import deepcopy
import time
from typing import Dict, Any, Iterator
from etils import epath

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from collections import defaultdict
import orbax.checkpoint as ocp

TrainState = Any


models = [
    'VAE',
    'TensorContracted',
    'TensorConFormer',
    'TensorConFormerEnc',
    'TensorConFormerDec',
    'Transformed',
    ]

class TrainerModule(object):
    '''
    Taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html
    and adapted as needed.
    '''
    def __init__(self,
                 model_class: nn.Module,
                 model_hparams: Dict,
                 optimizer_name: str,
                 optimizer_hparams: Dict,
                 checkpointer_params: Dict,
                 early_stopping_params: Dict,
                 exmp_input: Any,
                 dataset_name: str,
                 save_models: bool,
                 seed: int = 42,
                 debug: bool = False,
                 check_val_every_n_epoch: int = 1,
                 print_every_n_epochs: int = 1,
                 **kwargs):

        super().__init__()
        self.model_class = model_class
        self.optimizer_name = optimizer_name

        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.checkpointer_params = checkpointer_params
        self.early_stopping_params = early_stopping_params
        
        self.exmp_input = exmp_input
        self.seed = seed
        self.debug = debug
        self.dataset_name = dataset_name
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.save_models = save_models
        self.global_path = '/tmp/datasets/'
        self.print_every_n_epochs = print_every_n_epochs

        self.model = self.model_class(**self.model_hparams)
        self.create_jitted_functions()
        self.init_model(exmp_input)
        self.init_early_stopping()
        if self.save_models: self.init_checkpointer()

    def init_checkpointer(self):
        path = os.path.join(self.global_path, self.dataset_name, self.model_class.__name__)
        directory = epath.Path(path)
        if directory.exists():
            directory.rmtree()

        ckpt_params = deepcopy(self.checkpointer_params)
        save_interval_steps = ckpt_params.pop("save_interval_steps", 1)
        max_to_keep = ckpt_params.pop("max_to_keep", None)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep, 
            save_interval_steps=save_interval_steps,
            )
        
        self.mngr = ocp.CheckpointManager(
            directory=directory, 
            options=options,
            item_names=('state', 'train_metrics', 'val_metrics')
        )

    def init_model(self, exmp_input: Any):
        main_rng = random.key(self.seed)
        main_rng, model_rng  = random.split(main_rng)
        variables = self.run_model_init(exmp_input, model_rng)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables))
        print(f"Model {self.model_class.__name__} initialized with number of parameters = ", param_count)

        self.state = train_state.TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=variables,
            tx=None,
            opt_state=None,
        )
        self.rng = main_rng
    
    def run_model_init(self,
                       exmp_input: Any,
                       model_rng: jax.Array,
                       ) -> None:

        model_rng, init_rng, eps_rng = random.split(model_rng, num=3)
        if self.model_class.__name__ == 'VAE':
            return self.model.init(rngs={'params': init_rng, 'eps': eps_rng}, 
                                   x=exmp_input[0], c=exmp_input[1] if self.model.conditional else None)
        else:
            return self.model.init(rngs={'params': init_rng, 'eps': eps_rng}, 
                                   x=exmp_input[0], c=exmp_input[1] if self.model.conditional else None, training=False)

        
    def init_optimizer(self,
                       num_epochs: int,
                       num_steps_per_epoch: int) -> None:
        
        hparams = deepcopy(self.optimizer_hparams)

        lr = hparams.pop('lr', 1e-3)
        warmup = hparams.pop('warmup', 0)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=0.1 * lr
        )

        if self.optimizer_name == 'adam':
            transf = optax.clip_by_global_norm(hparams.pop('gradient_clip', 1.0))
            optimizer = optax.chain(
                transf,
                optax.adam(lr_schedule, **hparams)
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not found.")
        
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,                                           
            tx=optimizer,
        )
        
    def init_early_stopping(self) -> None:
        stop_params = deepcopy(self.early_stopping_params)
    
        min_delta = stop_params.pop('min_delta', 0.1)
        patience = stop_params.pop('patience', 10)
        best_metric = stop_params.pop('best_metric', float("inf")) ## monitor loss by default

        self.monitor = "acc" if best_metric == 0 else "loss"
        has_improved = True if self.monitor == "acc" else False
        self.early_stopping = EarlyStopping(
            min_delta=min_delta, 
            patience=patience, 
            best_metric=best_metric, 
            has_improved=has_improved
        )

    def create_jitted_functions(self):
        train_step, eval_step = self.create_functions()

        if self.debug:
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)
    
    def create_functions(self):
        
        def train_step(state: TrainState,
                       batch: Any,
                       batch_rng: jax.Array
                    ):
            metrics = {}
            return state, metrics
        def eval_step(state: TrainState,
                      batch: Any,
                      batch_rng: jax.Array
                    ):
            metrics = {}
            return metrics
        
        raise NotImplementedError
    
    def train_model(
            self,
            train_loader: Iterator,
            val_loader: Iterator,
            test_loader: Iterator = None,
            num_epochs: int = 20,
            save_best_models_only: bool = True
        ):
        
        evaluation_metrics = {
            'train': [],
            'val': [],
            'test': [] if test_loader else None,
        }
        self.init_optimizer(num_epochs, len(train_loader))
        self.rng, rng = random.split(self.rng)
        for epoch_idx in tqdm(range(1, num_epochs+1), desc='Epochs'):
            _, rng = random.split(rng)
            train_metrics = self.train_epoch(train_loader, rng)
            if epoch_idx % self.check_val_every_n_epoch == 0:
                val_metrics = self.eval_model(val_loader, rng)
                self.early_stopping = self.early_stopping.update(val_metrics[self.monitor])
                if self.early_stopping.has_improved:
                    self.early_stopping.reset()
                if epoch_idx % self.print_every_n_epochs == 0:
                    self.print_evaluation_metrics(val_metrics)
                if self.early_stopping.should_stop:
                    print(f"Met early stopping criteria, breaking at epoch {epoch_idx}.")
                    break
                if self.save_models:
                    if save_best_models_only and self.early_stopping.has_improved:
                        self.save_model(epoch_idx, train_metrics, val_metrics)
                    else:
                        self.save_model(epoch_idx, train_metrics, val_metrics)
            
            evaluation_metrics['train'].append(train_metrics)
            evaluation_metrics['val'].append(val_metrics)

        if self.save_models: self.mngr.wait_until_finished()
                
        if test_loader is not None:
            if self.save_models:
                self.load_model()
            test_metrics = self.eval_model(data_loader=test_loader, rng=rng)
            evaluation_metrics['test'].append(test_metrics)
        
        return evaluation_metrics
    
    def train_epoch(self, train_loader: Iterator, rng: jax.Array):
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        epoch_rng, batch_rng = random.split(rng)
        for batch in tqdm(train_loader, desc='Training', leave=False):
            _, batch_rng = random.split(batch_rng)
            self.state, step_metrics = self.train_step(self.state, batch, batch_rng)
            for key in step_metrics:
                metrics[key] += jnp.mean(step_metrics[key])
        
        metrics = {key: float(metrics[key] / num_train_steps) for key in metrics} # casting as float because orbax can't deal with jax.arrays in json dumps.
        metrics = {key: round(metrics[key], 5) for key in metrics}
        metrics['epoch_time'] = time.time() - start_time
        return metrics
    
    def eval_model(self, data_loader: Iterator, rng: jax.Array):
        metrics = defaultdict(float)
        num_val_steps = len(data_loader)
        epoch_rng, batch_rng = random.split(rng)
        for batch in data_loader:
            _, batch_rng = random.split(batch_rng)
            step_metrics = self.eval_step(self.state, batch, batch_rng)
            for key in step_metrics:
                metrics[key] += jnp.mean(step_metrics[key])
        
        metrics = {key: float(metrics[key] / num_val_steps) for key in metrics}
        metrics = {key: round(metrics[key], 5) for key in metrics}
        return metrics

    def save_model(self, step: int, train_metrics: dict, val_metrics: dict):
        self.mngr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(self.state),
                train_metrics=ocp.args.JsonSave(train_metrics),
                val_metrics=ocp.args.JsonSave(val_metrics)
            ),
        )

    def load_model(self, step: int = None):
        if step is None:
            step = self.mngr.latest_step()

        restored = self.mngr.restore(
            step=step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(),
                train_metrics=ocp.args.JsonRestore(),
                val_metrics=ocp.args.JsonRestore()
                )
            )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=restored.state['params'],
            tx=self.state.tx if self.state.tx else optax.sgd(0.1),
        )
    
    def load_from_checkpoint(self, step: int = None):
        if step is None:
            step = self.mngr.latest_step()

        restored = self.mngr.restore(
            step=step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(),
                train_metrics=ocp.args.JsonRestore(),
                val_metrics=ocp.args.JsonRestore()
                )
            )
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=restored.state['params'],
            tx=self.state.tx if self.state.tx else optax.sgd(0.1),
        )
        return state, restored.train_metrics, restored.val_metrics

    def print_evaluation_metrics(self, val_metrics):
            print("\t Loss: ", val_metrics['loss'])
