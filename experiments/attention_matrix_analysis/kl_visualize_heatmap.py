import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VisualizationEngine:
    @staticmethod
    def plot_results(model_key: str, task_name: str, head_matrix: np.ndarray, layer_array: np.ndarray,
                     output_root_dir: str):
        """Generate and save visualization charts."""
        save_dir = os.path.join(output_root_dir, model_key, task_name)
        os.makedirs(save_dir, exist_ok=True)

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

        # Head-wise KL Divergence Heatmap
        plt.figure(figsize=(14, 10))

        ax = sns.heatmap(
            head_matrix,
            cmap="Blues",
            annot=False,
            cbar_kws={'label': 'KL Divergence (Distribution Shift)'}
        )

        plt.title(f"Attention Head Shifts: {model_key} on {task_name}", fontsize=18, pad=20)
        plt.xlabel("Head Index", fontsize=14)
        plt.ylabel("Layer Index", fontsize=14)
        plt.gca().invert_yaxis()

        heatmap_path = os.path.join(save_dir, f"{model_key}_{task_name}_heatmap_head_kl.pdf")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved Heatmap to: {heatmap_path}")

        # Layer-wise Average KL Divergence Bar Plot
        plt.figure(figsize=(12, 6))

        layers = np.arange(len(layer_array))

        sns.barplot(x=layers, y=layer_array, color="#4c72b0", alpha=0.9)

        plt.title(f"Layer-wise Average Shift: {model_key} on {task_name}", fontsize=18, pad=20)
        plt.xlabel("Layer Index", fontsize=14)
        plt.ylabel("Avg KL Divergence", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Thin out X-axis labels if many layers
        if len(layers) > 20:
            for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
                if i % 2 != 0:
                    label.set_visible(False)

        barplot_path = os.path.join(save_dir, f"{model_key}_{task_name}_barplot_layer_kl.pdf")
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved Barplot to: {barplot_path}")

    @staticmethod
    def plot_model_summary(model_key: str, task_layer_means: dict, output_root_dir: str, data_root_dir: str = None):
        """
        Generate a cross-task layer-wise heatmap for a single model.
        X-axis: Tasks, Y-axis: Layers, Color: Average KL divergence per layer.
        """
        if not task_layer_means:
            logging.warning(f"No data found for model {model_key}, skipping summary plot.")
            return

        df_summary = pd.DataFrame(task_layer_means)
        df_summary = df_summary.reindex(sorted(df_summary.columns), axis=1)

        if data_root_dir:
            csv_save_dir = os.path.join(data_root_dir, model_key)
            os.makedirs(csv_save_dir, exist_ok=True)

            csv_path = os.path.join(csv_save_dir, f"{model_key}_layer_wise_summary.csv")
            df_summary.to_csv(csv_path, index=True)
            logging.info(f"Saved Summary CSV to: {csv_path}")

        save_dir = os.path.join(output_root_dir, model_key)
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

        ax = sns.heatmap(
            df_summary,
            cmap="Blues",
            cbar_kws={'label': 'Avg KL Divergence (Layer-wise)'}
        )

        plt.title(f"Layer-wise Shifts across Tasks: {model_key}", fontsize=16, pad=20)
        plt.xlabel("Task", fontsize=14)
        plt.ylabel("Layer", fontsize=14)

        plt.gca().invert_yaxis()
        plt.xticks(rotation=45, ha='right')

        summary_path = os.path.join(save_dir, f"{model_key}_ALL_TASKS_layer_heatmap.pdf")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved Model Summary Heatmap to: {summary_path}")


# Standalone mode: read from CSV and plot

if __name__ == "__main__":
    BASE_DIR = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/attention_matrix_analysis/old_attention_analysis_results/"
    OUTPUT_ROOT_FIGS = os.path.join(BASE_DIR, "figures")

    MODELS = ["gpt2", "llama2", "llama3", "qwen2"]
    TASKS = ["sentiment_yelp","sentiment_sst2","qa_squad", "qa_coqa", "mt_kde4", "mt_tatoeba"] #["sentiment_sst2-fix"]

    for TARGET_MODEL in MODELS:
        print(f"\n{'='*40}")
        print(f"Processing Model: {TARGET_MODEL}")
        print(f"{'='*40}")

        # Collect layer-wise means for all tasks under this model
        model_tasks_data = {}

        for TARGET_TASK in TASKS:
            csv_path = os.path.join(BASE_DIR, TARGET_MODEL, TARGET_TASK, "kl_divergence_heads.csv")

            try:
                if not os.path.exists(csv_path):
                    print(f"Skipping {TARGET_TASK}: File not found at {csv_path}")
                    continue

                print(f"Loading {TARGET_TASK}...")
                df = pd.read_csv(csv_path, index_col=0)

                head_matrix = df.values
                layer_mean = head_matrix.mean(axis=1)

                model_tasks_data[TARGET_TASK] = layer_mean

                VisualizationEngine.plot_results(
                    model_key=TARGET_MODEL,
                    task_name=TARGET_TASK,
                    head_matrix=head_matrix,
                    layer_array=layer_mean,
                    output_root_dir=OUTPUT_ROOT_FIGS
                )

            except Exception as e:
                print(f"Error processing {TARGET_TASK}: {e}")
                import traceback
                traceback.print_exc()

        # Generate summary heatmap across all tasks for this model
        if model_tasks_data:
            print(f"Generating Summary Heatmap for {TARGET_MODEL}...")
            VisualizationEngine.plot_model_summary(
                model_key=TARGET_MODEL,
                task_layer_means=model_tasks_data,
                output_root_dir=OUTPUT_ROOT_FIGS,
                data_root_dir=BASE_DIR
            )
        else:
            print(f"No valid data found for {TARGET_MODEL}, skipping summary.")

    print("\nAll Done!")
