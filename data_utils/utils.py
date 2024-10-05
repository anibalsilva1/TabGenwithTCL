import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from .dataloaders import create_data_loaders

def process_output(dataset: np.ndarray, 
                   meta: dict, 
                   transformations:dict):
    
    var_indices = meta['variable_indices']
    numerical_cols = meta['numerical_cols']
    categorical_cols = meta['categorical_cols']

    x_ = {}
    for var, idxs in var_indices.items():
        if np.size(idxs) == 1:
            x_[var] = dataset[:, idxs].squeeze()
        else:
            x_[var] = np.argmax(dataset[:, idxs], axis=-1)
    
    df = pd.DataFrame(x_)
    if numerical_cols and categorical_cols:
        numerical_transform, ordinal_transform = transformations['numerical'], transformations['ordinal']
        df[numerical_cols] = numerical_transform.inverse_transform(df[numerical_cols])
        df[categorical_cols] = ordinal_transform.inverse_transform(df[categorical_cols])
        
    elif not numerical_cols and categorical_cols:
        ordinal_transform = transformations['ordinal']
        df[categorical_cols] = ordinal_transform.inverse_transform(df[categorical_cols])
    
    elif numerical_cols and not categorical_cols:
        numerical_transform = transformations['numerical']
        df[numerical_cols] = numerical_transform.inverse_transform(df[numerical_cols])

    else:
        raise ValueError("Scenario not expected.")
    
    return df


def create_conditional_loaders(datasets, labels, batch_size):
    # pre-validation
    n_classes = jnp.array(jax.tree_util.tree_map(jnp.size, [jnp.unique(jnp.argmax(label, axis=-1)) for label in labels]))
    assert jnp.all(n_classes == n_classes[0]), 'Number of classes are not the same between datasets.'

    classes = jnp.unique(jnp.argmax(labels[0], axis=-1))
    
    cond_datasets = {}
    for cls in classes:
        indexed_datasets, indexed_labels = [], []
        for ds, lbl in zip(datasets, labels):
            labels_cat = jnp.argmax(lbl, axis=-1)
            indexes = jnp.where(labels_cat == cls)[0]

            indexed_datasets.append(ds[indexes])
            indexed_labels.append(lbl[indexes])

        cond_datasets[int(cls)] = create_data_loaders(
                dataframes=indexed_datasets, 
                labels=indexed_labels,
                is_train=[True, True, False],
                batch_size=batch_size)
    
    return cond_datasets