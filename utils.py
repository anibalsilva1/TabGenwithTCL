import pandas as pd
from typing import Callable
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from metrics import evaluate_highdensity, evaluate_ml_efficiency, evaluate_quality_report

eval_metrics = {
    'accuracy_syn_test': accuracy_score,
    'accuracy_train_test': accuracy_score,
    'fidelity': accuracy_score
    }


xgboost_hparams = {
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.9, 1],
    'colsample_bytree': [0.7, 0.9, 1],
    }

def evaluate_metric(
    metric_name: str,
    model_dir: str,
    dataset_name: str,
    df_train_real: pd.DataFrame,
    syn_df: pd.DataFrame,
    transformations: dict,
    meta: dict,
    df_test_real: pd.DataFrame=None):
    
    columns_name = ['dataset_name', 'model']
    row_values = [dataset_name, model_dir]

    if metric_name == 'highdensity':
        report = evaluate_highdensity(df_real=df_train_real, df_syn=syn_df, meta=meta, transformations=transformations)
        precision_alpha = report['delta_precision_alpha_naive']
        coverage_beta = report['delta_coverage_beta_naive']
        authenticity = report['authenticity_naive']

        columns_name = columns_name + ['alpha_precision', 'beta_recall', 'authenticity']
        row_values = row_values + [precision_alpha, coverage_beta, authenticity]
            
    elif metric_name == 'quality':
        df_train_real_ = df_train_real.copy()
        df_train_real_ = df_train_real_.drop(meta['target_name'], axis=1)
        syn_df_ = syn_df.copy()
        syn_df_ = syn_df_.drop(meta['target_name'], axis=1)
        
        report = evaluate_quality_report(df_real=df_train_real_, df_syn=syn_df_, meta=meta)
        
        marginal = report['marginal']
        pairs_corr = report['pairs_correlation']
        data_val = report['data_validity']
        data_struct = report['data_struct']
        
        columns_name = columns_name + ['marginal', 'pairs-correlation', 'data-validity', 'data-struct']
        row_values = row_values + [marginal, pairs_corr, data_val, data_struct]
    
    elif metric_name == "ml_efficiency":
        
        report = evaluate_ml_efficiency(
            real_train_df=df_train_real,
            real_test_df=df_test_real,
            syn_df=syn_df,
            model=XGBClassifier(),
            meta=meta,
            transformations=transformations,
            param_grid=xgboost_hparams,
            evaluation_metrics=eval_metrics
        )
        acc_syn_test = report['acc_syn_test'] 
        acc_train_test = report['acc_train_test']
        fidelity = report['fidelity']

        columns_name = columns_name + ['accuracy_syn_test', 'accuracy_train_test', 'fidelity']
        row_values = row_values + [acc_syn_test, acc_train_test, fidelity]

    else:
        raise ValueError(f"Metric {metric_name} not studied.")
    
    row = {k: v for k, v in zip(columns_name, row_values)}
    
    return row