import os
import pandas as pd
from tqdm import tqdm
import time
import jax
from jax import random
import pickle

from trainers import get_trainer

from data_utils import OpenMLCC18DataProcessor
from data_utils import create_data_loaders
from models import sample_from_model, get_model, estimate_batchsize

import warnings

from absl import flags
from absl import app
from absl import logging

logging.set_verbosity(logging.WARNING)

warnings.simplefilter(action='ignore')

FLAGS = flags.FLAGS

# general flags
flags.DEFINE_enum(name="model", 
                  default="TensorConFormer", 
                  enum_values=["VAE", "TensorContracted", "TensorConFormer", "Transformed", 
                               "TensorConFormerEnc", "TensorConFormerDec"], 
                  help="Model name.")

# training
flags.DEFINE_integer(name="print_every_n_epochs", default=15, help="Print evaluation loss every n epochs.")
flags.DEFINE_integer(name="num_epochs", default=300, help="Number of epochs to train.")
flags.DEFINE_boolean(name="save_models", default=False, help="Save models.")
flags.DEFINE_boolean(name="save_best_models_only", default=False, help="Save only best models.")
flags.DEFINE_integer(name="save_interval_steps", default=1, help="Save models between a given interval.")

# optimizer
flags.DEFINE_string(name="optimizer", default="adam", help="Optimizer name.")
flags.DEFINE_float(name="lr", default=1e-3, help="Learning rate.")
flags.DEFINE_integer(name="warmup", default=0, help="Learning rate warmup.")

# early stopping
flags.DEFINE_float(name="min_delta", default=1e-3, help="Improvement needed to not raise patience.")
flags.DEFINE_integer(name="patience", default=25, help="Patience.")
flags.DEFINE_string(name="best_metric", default="inf", help="Bound for best metric.")

def main(argv):
    synthethic_datasets_path = 'synthethic_datasets'
    metainfo_path = 'metainfo'

    save_syn_path = f'{synthethic_datasets_path}/{FLAGS.model}'
    if not os.path.exists(save_syn_path):
        os.makedirs(save_syn_path)

    save_metainfo_path = f'{metainfo_path}/{FLAGS.model}'
    if not os.path.exists(save_metainfo_path):
        os.makedirs(save_metainfo_path)
    

    cc18_meta = pd.read_json("cc18_metadata.json", orient="index")
    tasks_id = cc18_meta['task_id'].to_list()

    for i, task_id in tqdm(enumerate(tasks_id), desc="Dataset"):
        preprocessor = OpenMLCC18DataProcessor(numerical_transform='quantile')
        datasets, labels, transformations, meta = preprocessor.preprocessor(task_id=task_id)
        
        dataset_name = cc18_meta.loc[cc18_meta['task_id'] == task_id, 'dataset_name'].values[0]
        n_val = datasets[1].shape[0]
        batch_size = estimate_batchsize(n_val)
    
        train_loader, val_loader, test_loader = create_data_loaders(
            dataframes=datasets, 
            labels=labels, 
            is_train=[True, True, False], 
            batch_size=batch_size)
    
        variable_indices = meta['variable_indices']
    
        model_hparams = {}

        if FLAGS.model != 'VAE':
            output_dim = len(variable_indices)
            model_hparams['output_dim'] = output_dim
            model_hparams['variable_indices'] = variable_indices

        else:
            model_hparams['output_dim'] = datasets[0].shape[1]

        
        trainer_class = get_trainer(FLAGS.model)
    
        trainer = trainer_class(
            optimizer_name=FLAGS.optimizer,
            optimizer_hparams={'lr': FLAGS.lr,
                               'warmup': FLAGS.warmup},
            early_stopping_params={'min_delta': FLAGS.min_delta,
                                   'patience': FLAGS.patience,
                                   'best_metric': float(FLAGS.best_metric)},
            exmp_input=next(iter(train_loader)),
            dataset_name=dataset_name,
            checkpointer_params = {'save_interval_steps': FLAGS.save_interval_steps},
            save_models=FLAGS.save_models,
            print_every_n_epochs=FLAGS.print_every_n_epochs,
            model_hparams=model_hparams,
            indexes=variable_indices
        )

        start_training_time = time.time()
        metrics = trainer.train_model(train_loader=train_loader, 
                                      val_loader=val_loader, 
                                      test_loader=test_loader, 
                                      num_epochs=FLAGS.num_epochs, 
                                      save_best_models_only=FLAGS.save_best_models_only)
        training_time = time.time() - start_training_time

        start_sampling_time = time.time()
        df_syn = sample_from_model(trainer=trainer, 
                                   loader=train_loader, 
                                   n_samples=datasets[0].shape[0],
                                   seed=42,
                                   meta=meta,
                                   transformations=transformations)
        sampling_time = time.time() - start_sampling_time

        model_class = get_model(FLAGS.model)

        m = model_class(**model_hparams)
        params = m.init({'params': random.key(0), 'eps': random.key(1)}, next(iter(train_loader))[0], next(iter(train_loader))[1])
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))


        other_params = {}
        other_params['batch_size'] = batch_size
        other_params['training_time'] = training_time
        other_params['sampling_time'] = sampling_time
        other_params['num_parameters'] = param_count

        metainfo = {'model_hparams': model_hparams,
                    'other_params': other_params,
                    'metrics': metrics
                    }

        with open(f'{save_metainfo_path}/{dataset_name}.pkl', 'wb') as f:
            pickle.dump(metainfo, f)

        df_syn.to_csv(f'{save_syn_path}/{dataset_name}.csv', index=False)

if __name__ == "__main__":
    app.run(main)        

