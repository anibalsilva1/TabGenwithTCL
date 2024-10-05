import pandas as pd
import openml

def get_suite(suite_id):
    return openml.study.get_suite(suite_id=suite_id)

def get_tasks(suite):
    return suite.tasks

def get_task(task_id):
    return openml.tasks.get_task(task_id=task_id)

def get_dataset_meta_info(task):
    
    dataset_id = task.dataset_id
    target_name = task.target_name
    target_labels = task.class_labels

    dataset_meta = task.get_dataset()

    dataset_name = dataset_meta.name

    df, y, is_categorical, col_names = dataset_meta.get_data()
    categorical_columns = [name for name, cat in zip(col_names, is_categorical) if cat and name != target_name]
    numerical_columns = [name for name, cat in zip(col_names, is_categorical) if not cat and name != target_name]

    has_nans = df.isna().sum().sum()
    
    dataset_size = df.shape[0]
    num_features = df.shape[1]

    num_numerical = len(numerical_columns)
    num_categorical = len(categorical_columns)

    assert num_features == num_numerical + num_categorical + 1, 'Number of features wrongly determined.'

    task_type = 'binary' if len(target_labels) == 2 else 'multi-class'

    info = {'task_id': task.task_id,
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,
            'n_rows': dataset_size,
            'has_nans': has_nans,
            'target_name': target_name,
            'num_numerical': num_numerical,
            'num_categorical': num_categorical,
            'task_type': task_type
        }
    
    return info

def get_openmlcc18_datasets_meta_info(suite_id=99):
    suite = get_suite(suite_id=suite_id)
    tasks_id = get_tasks(suite)

    meta_info = {}
    for i, task_id in enumerate(tasks_id):
        task = get_task(task_id)
        dataset_id = task.dataset_id
        if dataset_id in [554, 40996, 40923, 40927] or task_id in [167125, 9910, 9976, 9981, 14970, 3481]: # image datasets and high-dim datasets
            continue
        meta_info[i] = get_dataset_meta_info(task)
    
    meta_info = pd.DataFrame(meta_info).T
    meta_info = meta_info.sort_values(by='n_rows').reset_index(drop=True)
    return meta_info