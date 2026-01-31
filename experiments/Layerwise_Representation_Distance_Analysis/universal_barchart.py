import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_distance_barchart(csv_path, task_name, model_name,output_filename=None):
    """Reads layer distance CSV and plots Before vs After bar chart."""
    print(f"Processing {task_name} from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    if 'layer' in df.columns and 'dist_before' in df.columns:
        df_avg = df.groupby('layer')[['dist_before', 'dist_after']].mean().reset_index()
    elif 'avg_dist_before' in df.columns:
        df_avg = df
    else:
        print("Error: CSV format not recognized. Columns found:", df.columns)
        return

    plt.figure(figsize=(12, 7))

    x = np.arange(len(df_avg['layer']))
    width = 0.35

    plt.bar(x - width/2, df_avg['avg_dist_before'], width, label='Before Finetuning', alpha=0.8, color='blue')
    plt.bar(x + width/2, df_avg['avg_dist_after'], width, label='After Finetuning', alpha=0.8, color='green')

    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Average Euclidean Distance', fontsize=14)
    if task_name== "Sentiment Analysis":
        title_content="Text-Label Distance Reduction"
    elif  task_name== "Question Answering":
        title_content="Query-Answer Distance Reduction"
    else:
        title_content="Source-Target Distance Reduction"
    plt.title(f'{model_name} {task_name}: {title_content} ', fontsize=16)
    plt.xticks(x, df_avg['layer'].astype(int))
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.2)

    #for i, row in df_avg.iterrows():
    #    change = ((row['dist_after'] - row['dist_before']) / row['dist_before']) * 100
    #    if row['dist_after'] > 0:
     #       plt.text(x[i] + width/2, row['dist_after'] + 0.5,
     #                f"{change:.1f}%", ha='center', va='bottom', fontsize=8, color='darkgreen')

    plt.tight_layout()

    if not output_filename:
        output_filename = f"{model_name}_{task_name.lower().replace(' ', '_')}_distance_chart.png"

    plt.savefig(output_filename, dpi=300)
    print(f"Chart saved to {output_filename}")
    # plt.show()

# Configuration
yelp_csv = r"<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/Results/Sentiment/Yelp/llama3/20260116_140525/filtered_layer_averages_sentiment.csv"
plot_distance_barchart(yelp_csv, "Sentiment Analysis","Llama-3.2-1B")

#squad_csv = r"./qa_analysis/YOUR_TIMESTAMP_FOLDER/all_layer_distances.csv"
# plot_distance_barchart(squad_csv, "Question Answering")

#mt_csv = r"./pairwise_distance_results/pairwise_distance_results.csv"
#if os.path.exists(mt_csv):
#    df_mt = pd.read_csv(mt_csv)
#    df_mt = df_mt.rename(columns={'avg_dist_before': 'dist_before', 'avg_dist_after': 'dist_after'})
#    df_mt.to_csv("temp_mt_formatted.csv", index=False)
#    plot_distance_barchart("temp_mt_formatted.csv", "Machine Translation")
#    os.remove("temp_mt_formatted.csv")
