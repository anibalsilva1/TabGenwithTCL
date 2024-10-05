import numpy as np
import pandas as pd
from typing import Union

from torch.utils import data

class TabularDataset(data.Dataset):

    def __init__(self, 
                 data: Union[pd.DataFrame, np.ndarray], 
                 label: Union[pd.DataFrame, np.ndarray]):
        self.data = data if isinstance(data, np.ndarray) else data.to_numpy()
        self.label = label if isinstance(label, np.ndarray) else label.to_numpy()
        self.n_classes = self.label.shape[1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]