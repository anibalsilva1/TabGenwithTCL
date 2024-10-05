import pandas as pd
import baycomp
import os
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

from absl import flags
from absl import app

plt.rcParams['text.usetex'] = True

FLAGS = flags.FLAGS

flags.DEFINE_enum("models", "main", ["main", "ablation"], "Models to evaluate.")


def main(argv):

    plots_path = 'plots'
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    eval_path = "evaluation_results"

    if FLAGS.models == "main":
        models = ['TensorConFormer', 'VAE', 'TensorContracted', 'Transformed']
        models_abvs = ['TCF', 'VAE', 'TC', 'TF']
        bbox_anchor = (-0.20, -0.25)

    elif FLAGS.models == "ablation":
        models = ["TensorConFormer", "TensorConFormerDec", "TensorConFormerEnc"]
        models_abvs = ["TCF", "TCFD", "TCFE"]
        bbox_anchor = (-0.10, -0.25)

    map_models = {k: v for k, v in zip(models, models_abvs)}
    
    pairs = list(itertools.combinations(models_abvs, 2))
    metrics = ['marginal', 'pairs-correlation', 'alpha_precision', 'beta_recall', 'accuracy_syn_test', 'fidelity']
    metrics_name = ['1-Way Marginals', 'Pairwise Corr.', '$\\alpha$-Precision', '$\\beta$-Recall', 'Utility', 'Fidelity']

    highdensity = pd.read_csv(f"{eval_path}/highdensity.csv")
    ml_eff = pd.read_csv(f"{eval_path}/ml_efficiency.csv")
    quality = pd.read_csv(f"{eval_path}/quality.csv")

    scores = highdensity.merge(ml_eff.merge(quality, 
                                            on=['dataset_name', 'model']), 
                                            on=['dataset_name', 'model']
                                            )

    ms = {k: v for k, v in zip(metrics, metrics_name)}
    ps = {'P_left': 'Left', 'P_draw': 'ROPE', 'P_right': 'Right'}
    
    scores['model'] = scores['model'].map(map_models)
    scores = scores.query('model in @models_abvs')

    new_model_scores = {}
    for model in models_abvs:
        metrics_res = {}
        for metric in metrics:
            metrics_res[metric] = scores.query('model == @model')[metric].to_numpy()

        new_model_scores[model] = metrics_res
    
    bst_pair_list = []
    for metric in metrics:
        for m1, m2 in pairs:
            metric_m1 = scores.query('model == @m1')[metric].to_numpy()
            metric_m2 = scores.query('model == @m2')[metric].to_numpy()
            
            bst = baycomp.SignTest.probs(x=metric_m1, y=metric_m2, rope=0.03, random_state=32)
            p_left = bst[0]
            p_draw = bst[1]
            p_right = bst[2]

            row = {
                'Pairs': m1 + " vs. " + m2,
                'Metric': metric,
                'P_left': p_left,
                'P_draw': p_draw,
                'P_right': p_right 
            }
            bst_pair_list.append(row)
    
    bst_pair = pd.DataFrame(bst_pair_list)
    bst_pair_melted = bst_pair.melt(id_vars=['Pairs', 'Metric'], 
                                    value_vars=['P_left', 'P_draw', 'P_right'], 
                                    var_name='Probabilities', value_name='Values')
    
    
    bst_pair_melted['Metric'] = bst_pair_melted['Metric'].map(ms)
    bst_pair_melted['Probabilities'] = bst_pair_melted['Probabilities'].map(ps)
    ordered_pairs = [f'{p[0]} vs. {p[1]}' for p in pairs]
    bst_pair_melted['Pairs'] = pd.Categorical(bst_pair_melted['Pairs'], categories=ordered_pairs[::-1], ordered=True)
    bst_pair_melted['Probabilities'] = pd.Categorical(bst_pair_melted['Probabilities'], categories=list(ps.values()), ordered=True)
    
    colormap = sns.color_palette("Set2")
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(100, 20), dpi=200)
    for i, metric in enumerate(metrics_name):
        bst_pair_melted.query('Metric == @metric') \
                       .pivot_table(index='Pairs', columns='Probabilities', values='Values') \
                       .plot(kind="barh", stacked=True, ax=axs[i], color=colormap)
        axs[i].set_xticklabels(axs[i].get_xticklabels(), size=50)
        axs[i].set_yticklabels(axs[i].get_yticklabels(), size=50)
        axs[i].set_xlabel("")
        axs[i].set_title(metric, size=70)
        axs[i].set_ylabel("")
        if i != 3:
            axs[i].get_legend().remove()
        else:
            axs[i].legend(loc='lower center', bbox_to_anchor=bbox_anchor, fontsize=55, ncols=3)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(f"{plots_path}/bayes_sign_test_{FLAGS.models}.pdf", format="pdf", dpi=150, bbox_inches="tight")
    

if __name__ == "__main__":
     app.run(main)