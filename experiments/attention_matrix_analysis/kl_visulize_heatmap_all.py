import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualizationEngine:
    # Academic mapping for X-axis task labels
    TASK_DISPLAY_MAP = {
        "sentiment_yelp": "Sentiment: Yelp",
        "sentiment_sst2": "Sentiment: SST-2",
        "qa_squad": "QA: SQuAD",
        "qa_coqa": "QA: CoQA",
        "mt_kde4": "MT: KDE4",
        "mt_tatoeba": "MT: Tatoeba"
    }

    @staticmethod
    def plot_combined_summary(all_models_data: dict, model_order: list, output_root_dir: str):
        """
        Generate a combined 1x4 heatmap figure for all models.
        Consistent physical height across models, single shared colorbar on the right.
        """
        sns.set_theme(style="white", context="paper")

        fig, axes = plt.subplots(1, 4, figsize=(24, 7.5))
        plt.subplots_adjust(wspace=0.15, bottom=0.25)

        sub_labels = ["(a)", "(b)", "(c)", "(d)"]

        for i, model_key in enumerate(model_order):
            if model_key not in all_models_data:
                continue

            ax = axes[i]
            df_summary = all_models_data[model_key]
            num_layers = len(df_summary)

            display_columns = [VisualizationEngine.TASK_DISPLAY_MAP.get(col, col)
                              for col in df_summary.columns]

            # Dynamic Y-axis step based on layer count
            if num_layers > 40: y_step = 10
            elif num_layers > 20: y_step = 5
            elif num_layers > 12: y_step = 2
            else: y_step = 1

            show_cbar = (i == len(model_order) - 1)
            sns.heatmap(
                df_summary,
                ax=ax,
                cmap="Blues",
                cbar=show_cbar,
                cbar_kws={'label': 'Avg KL Divergence'} if show_cbar else None,
                xticklabels=display_columns,
                yticklabels=y_step
            )

            ax.set_title(f"{sub_labels[i]} {model_key}", fontsize=22, pad=15, fontweight='bold')
            ax.set_xlabel("", labelpad=0)

            if i == 0:
                ax.set_ylabel("Layer Index", fontsize=18, fontweight='bold')
            else:
                ax.set_ylabel("")

            ax.tick_params(axis='y', labelleft=True, labelsize=14)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=14)

            ax.invert_yaxis()

        save_path = os.path.join(output_root_dir, "Figure_6_Clean_Xaxis.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Figure saved successfully to: {save_path}")


# Script Entry Point

if __name__ == "__main__":
    BASE_DIR = r"<PROJECT_ROOT>/experiments/attention_matrix_analysis/old_attention_analysis_results/"
    OUTPUT_ROOT_FIGS = os.path.join(BASE_DIR, "figures")
    os.makedirs(OUTPUT_ROOT_FIGS, exist_ok=True)

    MODELS = ["gpt2", "llama3", "qwen2", "llama2"]
    TASKS = ["sentiment_yelp", "sentiment_sst2", "qa_squad", "qa_coqa", "mt_kde4", "mt_tatoeba"]

    all_models_summary_data = {}
    for TARGET_MODEL in MODELS:
        model_tasks_data = {}
        for TARGET_TASK in TASKS:
            csv_path = os.path.join(BASE_DIR, TARGET_MODEL, TARGET_TASK, "kl_divergence_heads.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0)
                model_tasks_data[TARGET_TASK] = df.values.mean(axis=1)

        if model_tasks_data:
            df_model = pd.DataFrame(model_tasks_data)
            df_model = df_model.reindex(sorted(df_model.columns), axis=1)
            all_models_summary_data[TARGET_MODEL] = df_model

    if all_models_summary_data:
        VisualizationEngine.plot_combined_summary(all_models_summary_data, MODELS, OUTPUT_ROOT_FIGS)
