import jax

from typing import Union, Tuple, List, Iterator
import numpy as np
import pandas as pd

from torch.utils import data

from .datasets import TabularDataset

def numpy_collate(batch):
    return jax.tree_util.tree_map(np.asarray, data.default_collate(batch))

def create_data_loaders(
        dataframes: Union[Tuple[pd.DataFrame], Tuple[np.ndarray]], 
        labels: Union[Tuple[pd.Series], Tuple[np.ndarray]],
        is_train: List[bool],
        batch_size: int = 256,
        ) -> List[Iterator]:

    loaders = []
    for dataframe, label, is_train, in zip(dataframes, labels, is_train):
        dataset = TabularDataset(
            data=dataframe, 
            label=label,
        )

        loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,
            collate_fn=numpy_collate,
            drop_last=False,
        )
        loaders.append(loader)

    return loaders