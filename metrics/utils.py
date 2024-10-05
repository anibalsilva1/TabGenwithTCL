import numpy as np
import pandas as pd

from typing import Dict, Tuple, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

MLModel = Any

def meta_for_sdmetrics(meta: Dict):
    var_indices = meta['variable_indices']
    meta_sdv = {}
    meta_sdv['METADATA_SPEC_VERSION'] = 'SINGLE_TABLE_V1'

    columns = {}
    for var, idx in var_indices.items():
        if np.size(idx) > 1:
            columns[var] = {'sdtype': 'categorical'}
        else:
            columns[var] = {'sdtype': 'numerical'}
    
    meta_sdv['columns'] = columns
    return meta_sdv


def process_numerical_data(
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        numerical_cols: List,
    ) -> Tuple[np.ndarray, np.ndarray]:

    numerical_transform = StandardScaler()
    real = df_real.copy()
    syn = df_syn.copy()

    numerical_transform.fit(real[numerical_cols])
    x_real_num = numerical_transform.transform(real[numerical_cols])
    x_syn_num = numerical_transform.transform(syn[numerical_cols])

    return x_real_num, x_syn_num

def process_categorical_data(
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        categorical_cols: List,
        transformations: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    real = df_real.copy()
    syn  = df_syn.copy()
    ordinal_transform, categorical_transform = transformations['ordinal'], transformations['categorical']
    
    syn[categorical_cols] = ordinal_transform.transform(syn[categorical_cols])
    x_syn_cat = categorical_transform.transform(syn[categorical_cols])

    real[categorical_cols] = ordinal_transform.transform(real[categorical_cols])
    x_real_cat = categorical_transform.transform(real[categorical_cols])

    return x_real_cat, x_syn_cat

def process_data_for_evaluation(
        df_real: pd.DataFrame, 
        df_syn: pd.DataFrame, 
        meta: Dict, 
        transformations: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    numerical_cols = meta['numerical_cols']
    categorical_cols = meta['categorical_cols']

    if numerical_cols and categorical_cols:
        x_real_num, x_syn_num = process_numerical_data(df_real, df_syn, numerical_cols)
        x_real_cat, x_syn_cat = process_categorical_data(df_real, df_syn, categorical_cols, transformations)

        x_syn = np.concatenate([x_syn_num, x_syn_cat], axis=1)
        x_real = np.concatenate([x_real_num, x_real_cat], axis=1)
    
    elif not numerical_cols and categorical_cols:
        x_real, x_syn = process_categorical_data(df_real, df_syn, categorical_cols, transformations)
    
    elif numerical_cols and not categorical_cols:
        x_real, x_syn = process_numerical_data(df_real, df_syn, numerical_cols)
    
    return x_real, x_syn


def process_data_for_mle(
        x_real_train: pd.DataFrame, 
        x_real_test: pd.DataFrame, 
        x_syn: pd.DataFrame, 
        meta: Dict, 
        transformations: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    numerical_cols = meta['numerical_cols']
    categorical_cols = meta['categorical_cols']

    if numerical_cols and categorical_cols:
        numerical_transformation = StandardScaler()
        ordinal_transform, categorical_transform = transformations['ordinal'], transformations['categorical']

        numerical_transformation.fit(x_real_train[numerical_cols])
        
        x_train_num = numerical_transformation.transform(x_real_train[numerical_cols])
        x_test_num = numerical_transformation.transform(x_real_test[numerical_cols])
        x_syn_num = numerical_transformation.transform(x_syn[numerical_cols])

        x_syn[categorical_cols] = ordinal_transform.transform(x_syn[categorical_cols])
        x_syn_cat = categorical_transform.transform(x_syn[categorical_cols])

        x_real_train[categorical_cols] = ordinal_transform.transform(x_real_train[categorical_cols])
        x_train_cat = categorical_transform.transform(x_real_train[categorical_cols])

        x_real_test[categorical_cols] = ordinal_transform.transform(x_real_test[categorical_cols])
        x_test_cat = categorical_transform.transform(x_real_test[categorical_cols])

        x_train = np.concatenate([x_train_num, x_train_cat], axis=1)
        x_test = np.concatenate([x_test_num, x_test_cat], axis=1)
        x_syn = np.concatenate([x_syn_num, x_syn_cat], axis=1)
    
    elif not numerical_cols and categorical_cols:
        ordinal_transform, categorical_transform = transformations['ordinal'], transformations['categorical']

        x_syn[categorical_cols] = ordinal_transform.transform(x_syn[categorical_cols])
        x_syn = categorical_transform.transform(x_syn[categorical_cols])

        x_real_train[categorical_cols] = ordinal_transform.transform(x_real_train[categorical_cols])
        x_train = categorical_transform.transform(x_real_train[categorical_cols])

        x_real_test[categorical_cols] = ordinal_transform.transform(x_real_test[categorical_cols])
        x_test = categorical_transform.transform(x_real_test[categorical_cols])
    
    elif numerical_cols and not categorical_cols:

        numerical_transformation = StandardScaler()
        numerical_transformation.fit(x_real_train[numerical_cols])
        
        x_train = numerical_transformation.transform(x_real_train[numerical_cols])
        x_test = numerical_transformation.transform(x_real_test[numerical_cols])
        x_syn = numerical_transformation.transform(x_syn[numerical_cols])

    return x_train, x_test, x_syn


def grid_search(
        x: np.ndarray,
        y: np.ndarray,
        model: MLModel,
        param_grid: Dict, 
        score_fn: str, 
        seed: int,
        n_splits: int
    ) -> Dict:
    gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=score_fn,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            n_jobs=-1
        )

    gs.fit(x, y)
    best_params = gs.best_params_
    
    return best_params