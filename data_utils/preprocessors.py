import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from .utils import *
from .openmlcc18_utils import get_task

class OpenMLCC18DataProcessor(object):
    def __init__(self,
                 numerical_transform: str = 'quantile',
                 categorical_transform: str = 'one_hot',
                 nans_strategy: str = 'impute',
                 val_split_size: float = 0.15,
                 val_seed: int = 32,
                 as_numpy: bool = True
                 ):
        
        self.numerical_transform = numerical_transform
        self.categorical_transform = categorical_transform
        self.nans_strategy = nans_strategy
        self.val_split_size = val_split_size
        self.val_seed = val_seed
        self.as_numpy = as_numpy

    def get_train_test_split(self, task):
        return task.get_train_test_split_indices()
    
    def get_dataset(self, task, meta):
        target_name = meta['target_name']
        dataset = task.get_dataset()
        
        df, y, is_categorical, col_names = dataset.get_data()
        
        categorical_columns = [name for name, cat in zip(col_names, is_categorical) if cat and name != target_name]
        numerical_columns = [name for name, cat in zip(col_names, is_categorical) if not cat and name != target_name]
        return df, numerical_columns, categorical_columns

    def get_dataset_meta(self, task):
        dataset_id = task.dataset_id
        target_name = task.target_name
        target_labels = task.class_labels

        _, _, is_categorical, col_names = task.get_dataset().get_data()
        dataset_name = task.get_dataset().name

        task_type = 'binary' if len(target_labels) == 2 else 'multi-class'

        meta = {'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'target_name': target_name,
                'task_type': task_type,
                }
        return meta

    def impute(self, df, numerical_columns, categorical_columns):
        df_ = df.copy()
        if df_.isna().sum().sum() == 0:
            return df_, numerical_columns, categorical_columns
        
        # validate if there's any column with all NaNs:
        columns_with_all_nans = df_.columns[df_.isna().all()].to_list()
        if len(columns_with_all_nans) > 0:
            df_ = df_.drop(columns_with_all_nans, axis=1)
            numerical_columns = [num_col for num_col in numerical_columns if num_col not in columns_with_all_nans]
            categorical_columns = [cat_col for cat_col in categorical_columns if cat_col not in columns_with_all_nans]
    
        if self.nans_strategy == "impute":
            if categorical_columns:
                df_[categorical_columns] = SimpleImputer(strategy='most_frequent').fit_transform(df_[categorical_columns])
            if numerical_columns:
                df_[numerical_columns] = SimpleImputer(strategy='mean').fit_transform(df_[numerical_columns])
        elif self.nans_strategy == "drop":
            df_ = df_.dropna()
        else:
            raise NotImplementedError(f'NaNs strategy not found. Possible strategies: ["impute", "drop"].')
        return df_, numerical_columns, categorical_columns


    def drop_irrelevant_columns(self, df, numerical_columns, categorical_columns):
        '''Drops columns which have the same value for all rows.'''
        df_ = df.copy()
        if numerical_columns:
            stds = df_[numerical_columns].std(0)
            zero_std = np.where(stds == 0.0)[0].tolist()
            if zero_std:
                num_to_drop = [num_col for i, num_col in enumerate(numerical_columns) if i in zero_std]
                df_ = df_.drop(num_to_drop, axis=1)
                numerical_columns = [num_col for i, num_col in enumerate(numerical_columns) if i not in zero_std]
        
        if categorical_columns:
            unique_n_categories = df_[categorical_columns].nunique()
            unique_cat = np.where(unique_n_categories == 1)[0].tolist()
            if unique_cat:
                cats_to_drop = [cat_col for i, cat_col in enumerate(categorical_columns) if i in unique_cat]
                df_ = df_.drop(cats_to_drop, axis=1)
                categorical_columns = [cat_col for i, cat_col in enumerate(categorical_columns) if i not in unique_cat]
        
        return df_, numerical_columns, categorical_columns

    def split_train_test(self, X, y, train_idx, test_idx):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

        return X_train, y_train, X_test, y_test
    
    def split_train_val(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                          test_size=self.val_split_size, 
                                                          random_state=self.val_seed, 
                                                          stratify=y_train, shuffle=True)
        
        X_train, X_val = X_train.reset_index(drop=True), X_val.reset_index(drop=True)
        y_train, y_val = y_train.reset_index(drop=True), y_val.reset_index(drop=True)
        return X_train, y_train, X_val, y_val

    def get_numerical_transform(self, X_train, numerical_columns):
        if self.numerical_transform == 'std':
            normalizer = StandardScaler()
        elif self.numerical_transform == "minmax":
            normalizer = MinMaxScaler()
        elif self.numerical_transform == 'quantile':
            # Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
            normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=21)
        else:
            raise ValueError(f'Normalization type: {self.numerical_transform} not implemented. Possible normalizations: ["std", "minmax", "quantile"].')
        
        normalizer.fit(X_train[numerical_columns])
        return normalizer
    
    def ordinal_encoder(self, datasets, categorical_columns):
        X = pd.concat(datasets, axis=0)
        ordinal_transform = OrdinalEncoder()
        ordinal_transform.fit(X[categorical_columns])
        return ordinal_transform
    
    def get_categorical_transform(self, datasets, categorical_columns):
        '''
        For categorical variables, transformation is applied to the whole dataset 
        because some categories from test / val might not be present in train.
        '''
        X = pd.concat(datasets, axis=0)
        if self.categorical_transform == 'one_hot':
            categorical_transform = OneHotEncoder(sparse_output=False, 
                                                  drop=None,
                                                  handle_unknown='ignore')
        else:
            raise NotImplementedError(f'Categorical Transformation type: {self.categorical_transform} not implemented. Possible categorical transformation: "one_hot".')

        categorical_transform.fit(X[categorical_columns])
    
        return categorical_transform
        

    def transform_numerical(self, datasets, normalizer, numerical_columns):
        x_nums = []
        for dataset in datasets:
            x_num = normalizer.transform(dataset[numerical_columns])
            x_num = pd.DataFrame(x_num, columns=numerical_columns)
            x_nums.append(x_num)
        
        return x_nums
    
    def transform_categorical(self, datasets, cat_transform, categorical_columns):
        x_cats = []
        new_categories = cat_transform.get_feature_names_out()
        for dataset in datasets:
            x_cat = cat_transform.transform(dataset[categorical_columns])
            x_cat = pd.DataFrame(x_cat, columns=new_categories)
            x_cats.append(x_cat)

        return x_cats
    
    def transform_datasets(self, datasets, numerical_columns, categorical_columns):

        X_train, X_val, X_test = datasets
        transformations = {}
        if numerical_columns:
            numerical_normalizer = self.get_numerical_transform(X_train, numerical_columns)
            X_train_num, X_val_num, X_test_num = self.transform_numerical(datasets, numerical_normalizer, numerical_columns)
            transformations['numerical'] = numerical_normalizer

        if categorical_columns:
            ordinal_transform = self.ordinal_encoder(datasets, categorical_columns)
            for dataset in datasets:
                dataset[categorical_columns] = ordinal_transform.transform(dataset[categorical_columns])
                
            categorical_transform = self.get_categorical_transform(datasets, categorical_columns)
            X_train_cat, X_val_cat, X_test_cat = self.transform_categorical(datasets, categorical_transform, categorical_columns)
            transformations['ordinal'] = ordinal_transform
            transformations['categorical'] = categorical_transform
        
        if numerical_columns and categorical_columns:
            X_train = pd.concat([X_train_num, X_train_cat], axis=1)
            X_test = pd.concat([X_test_num, X_test_cat], axis=1)
            X_val = pd.concat([X_val_num, X_val_cat], axis=1)
            return (X_train, X_val, X_test), transformations
        
        elif numerical_columns and not categorical_columns:
            return (X_train_num, X_val_num, X_test_num), transformations
        
        elif not numerical_columns and categorical_columns:
            return (X_train_cat, X_val_cat, X_test_cat), transformations
        
        else:
            raise ValueError("Combination not expected.")

    def process_variable_indices(self, transformations, numerical_columns, categorical_columns):
        '''
        Processes the variable indices from the new dataset.
        Numerical variables always come first, if present.
        '''
        x_cat_start = 0
        if numerical_columns and categorical_columns:
            num_transform, cat_transform = transformations['numerical'], transformations['categorical']
            feats_num = num_transform.feature_names_in_

            numerical_indices = [np.array([i]) for i in range(len(feats_num))]
            numerical_indices = {k: v for k, v in zip(feats_num, numerical_indices)}
            x_cat_start = len(numerical_indices)

            feats_cat = cat_transform.feature_names_in_
            categories_sizes = [np.size(s) for s in cat_transform.categories_]
            cat_offsets = x_cat_start + np.cumsum(categories_sizes)
            
            categorical_indices = [np.arange(x_cat_start, cat_offsets[i]) if i == 0 else np.arange(cat_offsets[i-1], cat_offsets[i]) for i in range(len(cat_offsets))]
            categorical_indices = {k: v for k, v in zip(feats_cat, categorical_indices)}

            variable_indices = numerical_indices | categorical_indices
            return variable_indices
        
        elif numerical_columns and not categorical_columns:
            num_transform = transformations['numerical']
            
            feats_num = num_transform.feature_names_in_

            numerical_indices = [np.array([i]) for i in range(len(feats_num))]
            numerical_indices = {k: v for k, v in zip(feats_num, numerical_indices)}

            return numerical_indices
        
        elif not numerical_columns and categorical_columns:
            cat_transform = transformations['categorical']
            feats_cat = cat_transform.feature_names_in_
            categories_sizes = [np.size(s) for s in cat_transform.categories_]
            cat_offsets = np.cumsum(categories_sizes)
            cat_offsets = x_cat_start + cat_offsets

            category_indices = [np.arange(x_cat_start, cat_offsets[i]) if i == 0 else np.arange(cat_offsets[i-1], cat_offsets[i]) for i in range(len(cat_offsets))]
            category_indices = {k: v for k, v in zip(feats_cat, category_indices)}
            return category_indices
        
        else:
            raise ValueError("Combination not expected.")
    
    def label_encoder(self, labels):
        y_train = labels[0]
        label_encoder = OneHotEncoder(sparse_output=False)
        label_encoder.fit(y_train.to_frame())
        labels_oh = []
        for label in labels:
            label_oh = label_encoder.transform(label.to_frame())
            labels_oh.append(label_oh)
        return labels_oh, label_encoder

    def preprocessor(self, task_id):
        '''
        Main processor that encapsulates all the logic.
        '''
        task = get_task(task_id=task_id)
        train_idx, test_idx = self.get_train_test_split(task)
        meta = self.get_dataset_meta(task)
        df, numerical_columns, categorical_columns = self.get_dataset(task, meta)
        
        df_, numerical_columns, categorical_columns = self.impute(df=df, 
                                                                  numerical_columns=numerical_columns, 
                                                                  categorical_columns=categorical_columns)
        
        df_, numerical_columns, categorical_columns = self.drop_irrelevant_columns(df=df_, 
                                                                                   numerical_columns=numerical_columns, 
                                                                                   categorical_columns=categorical_columns)

        X, y = df_.drop(meta['target_name'], axis=1), df_[meta['target_name']]

        X_train, y_train, X_test, y_test = self.split_train_test(X, y, train_idx, test_idx)
        X_train, y_train, X_val, y_val = self.split_train_val(X_train, y_train)

        datasets = X_train, X_val, X_test
        labels = y_train, y_val, y_test
    
        datasets, transformations = self.transform_datasets(datasets=datasets, 
                                                            numerical_columns=numerical_columns, 
                                                            categorical_columns=categorical_columns)
        
        variable_indices = self.process_variable_indices(transformations=transformations, 
                                                         numerical_columns=numerical_columns, 
                                                         categorical_columns=categorical_columns)
        
        labels, label_encoder = self.label_encoder(labels)
        transformations['labels'] = label_encoder

        meta['variable_indices'] = variable_indices
        meta['categorical_cols'] = categorical_columns
        meta['numerical_cols'] =  numerical_columns
        meta['num_categorical'] = len(categorical_columns)
        meta['num_numerical'] = len(numerical_columns)
        
        if self.as_numpy:
            xs = []
            for dataset, label in zip(datasets, labels):
                xs.append(dataset.to_numpy())
                #ys.append(label.to_numpy())
            return xs, labels, transformations, meta
                
        
        return datasets, labels, transformations, meta