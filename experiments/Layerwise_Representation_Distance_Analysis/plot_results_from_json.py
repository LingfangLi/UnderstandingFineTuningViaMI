import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
RESULTS_DIR = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/Model_Internal_States_Analysis/Results/QA/SQUAD/llama3/20260116_142340/"

def main():
    print(f"Reading data from: {RESULTS_DIR}")

    json_files = glob.glob(os.path.join(RESULTS_DIR, "batch_*_summary.json"))

    if not json_files:
        print("Error: No batch summary JSON files found! Check your path.")
        return

    all_layer_data = []

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
            for sample in batch_data:
                for layer_info in sample['layer_distances']:
                    all_layer_data.append({
                        'layer': layer_info['layer'],
                        'dist_before': layer_info['dist_before'],
                        'dist_after': layer_info['dist_after']
                    })

    df = pd.DataFrame(all_layer_data)
    layer_avg = df.groupby('layer')[['dist_before', 'dist_after']].mean().reset_index()

    print("Data processed. Generating plot...")

    plt.figure(figsize=(14, 8))

    layers = layer_avg['layer']
    x = np.arange(len(layers))
    width = 0.35

    plt.bar(x - width/2, layer_avg['dist_before'], width, label='Before Fine-tuning', color='royalblue', alpha=0.85)
    plt.bar(x + width/2, layer_avg['dist_after'], width, label='After Fine-tuning', color='forestgreen', alpha=0.85)

    for i in range(len(layers)):
        before = layer_avg.iloc[i]['dist_before']
        after = layer_avg.iloc[i]['dist_after']
        pct_change = ((after - before) / before) * 100

        if pct_change < -5:
            plt.text(i, max(before, after) + 1, f"{pct_change:.1f}%",
                     ha='center', color='darkred', fontsize=9, fontweight='bold')

    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Euclidean Distance (Source vs Target)', fontsize=14)
    plt.title('Layer-wise Representation Distance: Before vs After Fine-tuning', fontsize=16)
    plt.xticks(x, layers)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "distance_comparison_chart.png")
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved successfully to: {save_path}")

    csv_save_path = os.path.join(RESULTS_DIR, "layer_average_distances.csv")
    layer_avg.to_csv(csv_save_path, index=False)
    print(f"Averaged data saved to: {csv_save_path}")

if __name__ == "__main__":
    main()
