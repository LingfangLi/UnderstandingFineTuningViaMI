import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
csv_path = "/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/cross_task_edges/summary_tables/Summary_Task2_SourceFT_vs_TargetPT_Overlap_Percentages.csv"
output_dir = "/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/cross_task_edges/summary_tables/"
os.makedirs(output_dir, exist_ok=True)

# Axis ordering
TASK_ORDER_X = ["coqa", "kde4", "squad", "sst2", "tatoeba", "yelp"]
TASK_ORDER_Y = TASK_ORDER_X[::-1]

# Display names
DISPLAY_NAMES = {
    "yelp": "Yelp", "sst2": "SST-2", 
    "coqa": "CoQA", "squad": "SQuAD", 
    "kde4": "KDE4", "tatoeba": "Tatoeba"
}

MODELS_ORDER = ["gpt2", "llama3.2", "qwen2", "llama2"]

# Data loading
if not os.path.exists(csv_path):
    logging.error(f"CSV file not found at: {csv_path}")
    exit()

df_raw = pd.read_csv(csv_path, index_col=[0, 1])

# Global color scale across all models
all_values = df_raw.values.flatten()
global_vmin = np.nanmin(all_values)
global_vmax = np.nanmax(all_values)

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_final_academic_fig5(df, model_order, output_path):
    """Generate a 1x4 heatmap grid with uniform sizing and shared colorbar."""
    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="white", context="paper")
    
    fig, axes = plt.subplots(1, 4, figsize=(26, 8))
    plt.subplots_adjust(wspace=0.2, bottom=0.3, left=0.12, right=0.9)

    sub_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, model in enumerate(model_order):
        ax = axes[i]
        try:
            data = df.xs(model, level='Model_Arch').T
            data = data.reindex(index=TASK_ORDER_Y, columns=TASK_ORDER_X)
            
            data.index = [DISPLAY_NAMES.get(n, n) for n in data.index]
            data.columns = [DISPLAY_NAMES.get(n, n) for n in data.columns]

            sns.heatmap(
                data,
                ax=ax,
                cmap="Blues",
                square=True,
                vmin=global_vmin,
                vmax=global_vmax,
                cbar=False,
                mask=data.isnull()
            )

            ax.set_title(f"{sub_labels[i]} {model}", fontsize=22, fontweight='bold', pad=20)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=14)
            plt.setp(ax.get_yticklabels(), rotation=45, va="top", fontsize=14)

        except Exception as e:
            logging.error(f"Error processing {model}: {e}")
            ax.set_visible(False)

    # Standalone colorbar aligned to the last subplot
    last_ax_pos = axes[3].get_position()
    cax = fig.add_axes([0.92, last_ax_pos.y0, 0.015, last_ax_pos.height])
    

    norm = Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = ScalarMappable(norm=norm, cmap="Blues")
    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label('Overlap Percentage (%)', fontsize=16, fontweight='bold')
    cax.tick_params(labelsize=12)

    save_file = os.path.join(output_path, "Figure_5_Standardized.pdf")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Refined Figure 5 saved to: {save_file}")

# Execute
plot_final_academic_fig5(df_raw, MODELS_ORDER, output_dir)