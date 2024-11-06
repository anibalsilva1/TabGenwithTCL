import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from absl import flags
from absl import app
from absl import logging

from data_utils import OpenMLCC18DataProcessor
from data_utils import process_output
from utils import evaluate_metric

logging.set_verbosity(logging.WARNING)
FLAGS = flags.FLAGS
flags.DEFINE_enum(name="metric_name", default="highdensity", enum_values=["highdensity", "ml_efficiency", "quality"], help="Metric name.")

def main(argv):
    metrics_res_path = "evaluation_results"
    if not os.path.exists(metrics_res_path):
        os.mkdir(metrics_res_path)

    cc18_meta = pd.read_json("cc18_metadata.json", orient="index")
    
    syn_csvs_dir = "synthethic_datasets"
    models_dir = ["VAE", "TensorContracted", "TensorConFormer", 
                  "Transformed", "TensorConFormerEnc", "TensorConFormerDec"
                  ]

    res = []
    for i in tqdm(range(len(cc18_meta)), desc="Evaluation datasets: "):

        task_id = int(cc18_meta.loc[i, 'task_id'])
        dataset_name = cc18_meta.loc[i, 'dataset_name']
        preprocessor = OpenMLCC18DataProcessor(numerical_transform='quantile')
        datasets, labels, transformations, meta = preprocessor.preprocessor(task_id=task_id)

        df_train_real = process_output(dataset=datasets[0], meta=meta, transformations=transformations)
        df_train_real[meta['target_name']] = np.argmax(labels[0], axis=-1)
        df_test_real = process_output(dataset=datasets[2], meta=meta, transformations=transformations)
        df_test_real[meta['target_name']] = np.argmax(labels[2], axis=-1)
    
        for model_dir in models_dir:
    
            path = os.path.join(syn_csvs_dir, model_dir)
            df_path = path + "/" + dataset_name + ".csv"
            syn_df = pd.read_csv(df_path, dtype=df_train_real.dtypes.to_dict())

            row = evaluate_metric(
                metric_name=FLAGS.metric_name,
                model_dir=model_dir,
                dataset_name=dataset_name,
                df_train_real=df_train_real,
                syn_df=syn_df,
                transformations=transformations,
                meta=meta,
                df_test_real=df_test_real)


            res.append(row)

    df = pd.DataFrame(res)
    df.to_csv(f"{metrics_res_path}/{FLAGS.metric_name}.csv", index=False)

if __name__ == "__main__":
    app.run(main)


