import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def make_radar_plot(df, plots_path):

    rows_by = ['Dataset Size', 'Feature Size']
    cols_by = ['$\\alpha$-Precision', '$\\beta$-Recall', 'Fidelity', 'Utility', 'Mean']

    colors = sns.color_palette("Set2")
    
    models = ["VAE", "TensorContracted", "Transformed", "TensorConFormer"]
    df = df.query("model in @models")
    fig, axs = plt.subplots(nrows=2, ncols=5, subplot_kw=dict(polar=True), figsize=(16, 10))  # Increase the height to 10
    for i, row in enumerate(rows_by):
        num_angles = len(df[row].unique())
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).tolist()
        angles += angles[:1]

        for j, col in enumerate(cols_by):

            if col == "Mean":
                melted_metric_results = df.melt(id_vars=['dataset_name', 'model', row], 
                                                             value_vars=cols_by[:-1], 
                                                             value_name='scores', var_name='metric')

                melted_metric_results['ranks'] = melted_metric_results.groupby(['dataset_name', 'metric'], observed=True)['scores'].rank(ascending=False)
                melted_metric_results_mean = melted_metric_results.groupby([row, 'model'], observed=True)['ranks'].mean().reset_index()
                df_pivot = melted_metric_results_mean.pivot_table(index=row, columns='model', values='ranks', observed=True)

            else:
                df_ = df.copy()
                df_[f"Rank {row}"] = df_.groupby(['dataset_name'], observed=True)[col].rank(ascending=False).astype(int)
                average_ranks = df_.groupby([row, 'model'], observed=True)[f'Rank {row}'].mean().reset_index()
                df_pivot = average_ranks.pivot(index=row, columns='model', values=f'Rank {row}')
            
            df_pivot = df_pivot[models]

            for k, model in enumerate(df_pivot.columns):
                
                values = df_pivot[model].tolist()
                values += values[:1]

                axs[i, j].plot(angles, values, label=model, color=colors[k])
                axs[i, j].fill(angles, values, alpha=0.2, color=colors[k])
                

            axs[i, j].set_xticks(angles[:-1])
            axs[i, j].set_xticklabels(df_pivot.index.tolist())
            axs[i, j].set_yticks([1, 2, 3, 4])
            axs[i, j].set_yticklabels([1, 2, 3, 4])
            axs[i, j].tick_params(axis='x', pad=10)

            if i == 0:
                axs[i, j].set_title(col, size=25, pad=50)

            if j == 0:
                axs[i, j].text(-0.5, 0.9, f'{row}', transform=axs[i, j].transAxes, fontsize=30, verticalalignment='top', rotation=90)
        
    plt.tight_layout()  # Adjust layout to avoid legend overlap
    plt.legend(loc='lower center', bbox_to_anchor=(-2.5, -0.5), ncol=len(models), fontsize=13)  # Adjust bbox_to_anchor to (0.5, -0.2)
    plt.savefig(f"{plots_path}/radar.pdf", format="pdf", dpi=200, bbox_inches="tight")    