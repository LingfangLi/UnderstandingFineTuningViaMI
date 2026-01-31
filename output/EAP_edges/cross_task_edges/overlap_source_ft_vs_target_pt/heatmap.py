import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
csv_path = "<PROJECT_ROOT>/output/EAP_edges/cross_task_edges/summary_tables/Summary_Task2_SourceFT_vs_TargetPT_Overlap_Percentages.csv"
TARGET_MODEL = "all"  # "all" or specific model name, e.g. "gpt2", "llama2"

# Read data
if not os.path.exists(csv_path):
    print(f"Error: File not found {csv_path}")
    exit()

df = pd.read_csv(csv_path, index_col=[0, 1])
all_models = df.index.get_level_values('Model_Arch').unique().tolist()

def plot_model_heatmap(model_name, full_df, output_dir):
    print(f"Generating heatmap for: {model_name}...")
    
    try:
        data_matrix = full_df.xs(model_name, level='Model_Arch')
    except KeyError:
        print(f"Error: Model '{model_name}' not found in data.")
        return

    # Y-axis: Pretrained, X-axis: Fine-tuned
    plot_data = data_matrix.T
    plot_data.index = [f"Pretrained {str(name).capitalize()}" for name in plot_data.index]
    plot_data.columns = [f"Fine-tuned {str(name).capitalize()}" for name in plot_data.columns]

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        plot_data,
        #annot=True,            # Change to True if percentage numbers need to be displayed
        #fmt=".1f",             # Keep one decimal place for numbers
        cmap="Blues",
        square=True,
        cbar_kws={'label': 'Overlap Percentage (%)'},
        xticklabels=True,
        yticklabels=True
    )

    plt.title(f"Edge Overlap: {model_name}\nPT models vs FT models", fontsize=14, fontweight='bold')
    plt.ylabel("Pretrained Models (PT)", fontsize=12, labelpad=10)
    plt.xlabel("Fine-tuned Models (FT)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{model_name}_heatmap.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to: {save_path}")

output_dir = os.path.dirname(csv_path)

if TARGET_MODEL == "all":
    print(f"Mode: Processing all models {all_models}")
    for model in all_models:
        plot_model_heatmap(model, df, output_dir)
else:
    if TARGET_MODEL in all_models:
        plot_model_heatmap(TARGET_MODEL, df, output_dir)
    else:
        print(f"Error: Target model '{TARGET_MODEL}' not found. Available models: {all_models}")

print("\nProcessing complete.")