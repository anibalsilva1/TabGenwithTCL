import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import make_radar_plot

plt.rcParams['text.usetex'] = True

metrics_names = {
    'accuracy_syn_test': 'Utility',
    'fidelity': 'Fidelity',
    'alpha_precision': '$\\alpha$-Precision',
    'beta_recall': '$\\beta$-Recall'
}

def main():
    plots_path = 'plots'
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    cc18_meta = pd.read_json("cc18_metadata.json", orient="index")
    cc18_meta['num_features'] = cc18_meta['num_categorical'] + cc18_meta['num_numerical']
    cc18_meta_cols = ['dataset_name', 'n_rows', 'num_features']
    cc18_meta = cc18_meta.loc[:, cc18_meta_cols]

    dataset_size_bins = [0, 0.8e3, 1.5e3, 3e3, 5e3, 10e3, 10e4]
    features_size_bins = [0, 5, 10, 20, 30, 50, 100, 1000]

    dataset_size_strs = ["$<$ 0.8", "[0.8, 1.5)", "[1.5, 3)", "[3, 5)", "[5, 10)", "$\\geq$ 10"]
    features_size_strs = ["$<$ 5", "[5, 10)", "[10, 20)", "[20, 30)", "[30, 50)", "[50, 100)", "$\\geq$ 100"]
    
    cc18_meta['Dataset Size'] = pd.cut(cc18_meta['n_rows'], bins=dataset_size_bins, labels=dataset_size_strs)
    cc18_meta['Feature Size'] = pd.cut(cc18_meta['num_features'], bins=features_size_bins, labels=features_size_strs)
    
    highdensity = pd.read_csv("evaluation_results/highdensity.csv")
    ml_efficiency = pd.read_csv("evaluation_results/ml_efficiency.csv")
    quality = pd.read_csv("evaluation_results/quality.csv")
    
    metrics_results = ml_efficiency.merge(highdensity.merge(quality, on=['dataset_name', 'model']), on=['dataset_name', 'model'])
    metrics = ['alpha_precision', 'beta_recall', 'fidelity', 'accuracy_syn_test', 'Mean']
    heads = ['dataset_name', 'model']
    
    metrics_results = metrics_results.loc[:, heads + metrics[:-1]]
    metrics_results = metrics_results.merge(cc18_meta, on="dataset_name")
    metrics_results = metrics_results.rename(metrics_names, axis=1)

    make_radar_plot(df=metrics_results, plots_path=plots_path)

if __name__ == "__main__":
    main()