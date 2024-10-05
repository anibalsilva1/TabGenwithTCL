import numpy as np
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split

from .generation import *

class SyntheticDatasets(object):
    def __init__(self,
                 syn_df: str,
                 n_samples: int = 20_000,
                 noise: float = 0.0,
                 val_size: float = 0.15,
                 test_size: float = 0.2,
                 ):
        
        self.n_samples = n_samples
        self.noise = noise
        self.syn_df = syn_df
        self.val_size = val_size
        self.test_size = test_size


    def split_train_test_val(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=32, shuffle=True, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=3, shuffle=True, stratify=y_train)

        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def transform(self, datasets, labels):
        X_train = datasets[0]
        y_train = labels[0]

        normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=21)
        oh = OneHotEncoder(sparse_output=False)
        normalizer.fit(X_train)
        oh.fit(y_train[..., None])

        Xs = [normalizer.transform(dataset) for dataset in datasets]
        ys = [oh.transform(y[..., None]) for y in labels]

        transformations = {'quantile': normalizer, 'ohe': oh}

        return Xs, ys, transformations
    
    def processor(self):
        if self.syn_df == 'flowers':
            X, y = generate_flower_data(N=self.n_samples, a=3.5, b=2.5, radius=1, noise_std=self.noise, two_class=False)
        elif self.syn_df == 'sin':
            X, y = generate_sin_wave_data(N=self.n_samples, noise_std=self.noise, k=5)
        elif self.syn_df == 'kite':
            X, y = generate_kite_data(N=self.n_samples, noise_std=self.noise)
        elif self.syn_df == 'circles':
            X, y = generate_circle_data(N=self.n_samples, radius=1, noise_std=self.noise)
        elif self.syn_df == 'xor':
            X, y = generate_xor_data(N=self.n_samples)
        elif self.syn_df == 'moon':
            X, y = generate_moon_data(n_class1=int(self.n_samples / 2), n_class2=int(self.n_samples / 2))
        elif self.syn_df == 'hourglass':
            X, y = generate_hourglass_data(N=self.n_samples, noise_std=self.noise)
        else:
            raise ValueError(f"Dataset: {self.syn_df} not studied.")
        
        datasets, labels = self.split_train_test_val(X=X, y=y)
        datasets, labels, transformations = self.transform(datasets=datasets, labels=labels)
        variable_indices = {'x'+str(i): np.array([i]) for i in range(datasets[0].shape[1])}

        return datasets, labels, transformations, variable_indices