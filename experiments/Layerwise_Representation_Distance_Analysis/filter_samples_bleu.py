import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from datasets import load_dataset

def load_batch_results(output_dir):
    """Load all batch summary JSON files."""
    all_samples = []

    json_files = sorted(glob(os.path.join(output_dir, "batch_*_summary.json")))

    print(f"Found {len(json_files)} batch files")

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
            all_samples.extend(batch_data)

    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples


def filter_samples(all_samples, min_decreased_ratio=0.5, require_bleu_improvement=True,require_average_distance_reduction=False):
    """
    Filter samples meeting specified criteria.

    Parameters:
    - min_decreased_ratio: minimum fraction of layers with decreased distance (default 0.5)
    - require_bleu_improvement: whether BLEU score must improve
    - require_average_distance_reduction: whether average distance must decrease (default False)
    """
    filtered_samples = []

    for sample in all_samples:
        bleu_improved = sample['bleu_after'] > sample['bleu_before']

        total_layers = len(sample['layer_distances'])
        decreased_layers = sum(1 for layer in sample['layer_distances'] if layer['is_decreased'])
        decreased_ratio = decreased_layers / total_layers

        avg_dist_before = np.mean([layer['dist_before'] for layer in sample['layer_distances']])
        avg_dist_after = np.mean([layer['dist_after'] for layer in sample['layer_distances']])
        avg_dist_change = avg_dist_after - avg_dist_before

        if avg_dist_change < 0:
            sample['avg_dist_change'] = avg_dist_change
            distance_reduced = True
        else:
            sample['avg_dist_change'] = 0
            distance_reduced = False

        meets_layer_criteria = decreased_ratio >= min_decreased_ratio
        meets_bleu_criteria = bleu_improved if require_bleu_improvement else True
        meet_distance_criteria = distance_reduced if require_average_distance_reduction else True

        if meets_layer_criteria and meets_bleu_criteria and meet_distance_criteria:
            sample['decreased_layers'] = decreased_layers
            sample['decreased_ratio'] = decreased_ratio
            filtered_samples.append(sample)

    print(f"\nFiltering results:")
    print(
        f"  - Samples with >{min_decreased_ratio * 100}% layers decreased: {len([s for s in all_samples if sum(1 for l in s['layer_distances'] if l['is_decreased']) / len(s['layer_distances']) >= min_decreased_ratio])}")
    print(f"  - Samples with BLEU improvement: {len([s for s in all_samples if s['bleu_after'] > s['bleu_before']])}")
    print(f"  - Samples meeting both criteria: {len(filtered_samples)}")

    return filtered_samples


def calculate_layer_averages(filtered_samples):
    """Compute per-layer averages across filtered samples."""
    if not filtered_samples:
        print("No samples to analyze!")
        return None

    n_layers = len(filtered_samples[0]['layer_distances'])

    layer_stats = {i: {
        'dist_before_sum': 0,
        'dist_after_sum': 0,
        'dist_change_sum': 0,
        'change_percent_sum': 0,
        'decreased_count': 0,
        'sample_count': 0
    } for i in range(n_layers)}

    for sample in filtered_samples:
        for layer_data in sample['layer_distances']:
            layer_idx = layer_data['layer']
            layer_stats[layer_idx]['dist_before_sum'] += layer_data['dist_before']
            layer_stats[layer_idx]['dist_after_sum'] += layer_data['dist_after']
            layer_stats[layer_idx]['dist_change_sum'] += layer_data['dist_change']
            layer_stats[layer_idx]['change_percent_sum'] += layer_data['change_percent']
            layer_stats[layer_idx]['decreased_count'] += int(layer_data['is_decreased'])
            layer_stats[layer_idx]['sample_count'] += 1

    layer_averages = []
    for layer_idx in range(n_layers):
        stats = layer_stats[layer_idx]
        n = stats['sample_count']

        layer_avg = {
            'layer': layer_idx,
            'avg_dist_before': stats['dist_before_sum'] / n,
            'avg_dist_after': stats['dist_after_sum'] / n,
            'avg_dist_change': stats['dist_change_sum'] / n,
            'avg_change_percent': stats['change_percent_sum'] / n,
            'decreased_percentage': (stats['decreased_count'] / n) * 100
        }
        layer_averages.append(layer_avg)

    return pd.DataFrame(layer_averages)


def analyze_filtered_samples(filtered_samples):
    """Print analysis of filtered samples."""
    print("\n" + "=" * 60)
    print("ANALYSIS OF FILTERED SAMPLES")
    print("=" * 60)

    print(f"\nTotal filtered samples: {len(filtered_samples)}")

    bleu_before = [s['bleu_before'] for s in filtered_samples]
    bleu_after = [s['bleu_after'] for s in filtered_samples]
    bleu_improvements = [s['bleu_improvement'] for s in filtered_samples]

    print(f"\nBLEU Score Statistics:")
    print(f"  Average BLEU before: {np.mean(bleu_before):.4f}")
    print(f"  Average BLEU after: {np.mean(bleu_after):.4f}")
    print(f"  Average BLEU improvement: {np.mean(bleu_improvements):.4f}")
    print(f"  Max BLEU improvement: {np.max(bleu_improvements):.4f}")

    decreased_ratios = [s['decreased_ratio'] for s in filtered_samples]
    print(f"\nLayer Decrease Statistics:")
    print(f"  Average ratio of layers with decrease: {np.mean(decreased_ratios):.2%}")
    print(f"  Samples with ALL layers decreased: {sum(1 for r in decreased_ratios if r == 1.0)}")

    print("\nTop 5 samples by BLEU improvement:")
    sorted_samples = sorted(filtered_samples, key=lambda x: x['bleu_improvement'], reverse=True)
    for i, sample in enumerate(sorted_samples[:5]):
        print(f"\n  Sample {i + 1} (idx: {sample['sample_idx']}):")
        print(f"    Source: {sample['en'][:50]}...")
        print(f"    Target: {sample['fr'][:50]}...")
        print(f"    BLEU: {sample['bleu_before']:.4f} -> {sample['bleu_after']:.4f} (+{sample['bleu_improvement']:.4f})")
        print(f"    Layers decreased: {sample['decreased_layers']}/{len(sample['layer_distances'])}")


def visualize_layer_averages(layer_averages_df, output_dir):
    """Plot layer-wise analysis for filtered MT samples."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    x = layer_averages_df['layer']
    ax1.plot(x, layer_averages_df['avg_dist_before'], 'b-', label='Before fine-tuning', linewidth=2)
    ax1.plot(x, layer_averages_df['avg_dist_after'], 'r-', label='After fine-tuning', linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Distance')
    ax1.set_title('Average Distance by Layer (Filtered Samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    colors = ['red' if x < 0 else 'blue' for x in layer_averages_df['avg_change_percent']]
    ax2.bar(x, layer_averages_df['avg_change_percent'], color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Average Change (%)')
    ax2.set_title('Average Distance Change Percentage by Layer')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.bar(x, layer_averages_df['decreased_percentage'])
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Percentage of Samples (%)')
    ax3.set_title('Percentage of Samples with Distance Decrease by Layer')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.bar(x, layer_averages_df['avg_dist_change'],
            color=['red' if x < 0 else 'blue' for x in layer_averages_df['avg_dist_change']])
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Average Distance Change')
    ax4.set_title('Average Absolute Distance Change by Layer')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtered_layer_analysis.png'), dpi=300)
#    plt.show()


def save_filtered_results(filtered_samples, layer_averages_df, output_dir):
    """Save filtered results to JSON, CSV files."""
    filtered_json_path = os.path.join(output_dir, 'filtered_samples.json')
    with open(filtered_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_samples, f, ensure_ascii=False, indent=2)

    layer_avg_csv_path = os.path.join(output_dir, 'filtered_layer_averages.csv')
    layer_averages_df.to_csv(layer_avg_csv_path, index=False)

    sample_details = []
    for sample in filtered_samples:
        detail = {
            'sample_idx': sample['sample_idx'],
            'source': sample['en'],
            'target': sample['fr'],
            'prediction_before': sample['prediction_before'],
            'prediction_after': sample['prediction_after'],
            'bleu_before': sample['bleu_before'],
            'bleu_after': sample['bleu_after'],
            'bleu_improvement': sample['bleu_improvement'],
            'decreased_layers': sample['decreased_layers'],
            'decreased_ratio': sample['decreased_ratio']
        }
        sample_details.append(detail)

    details_df = pd.DataFrame(sample_details)
    details_csv_path = os.path.join(output_dir, 'filtered_sample_details.csv')
    details_df.to_csv(details_csv_path, index=False, encoding='utf-8')

    print(f"\nResults saved to:")
    print(f"  - Filtered samples: {filtered_json_path}")
    print(f"  - Layer averages: {layer_avg_csv_path}")
    print(f"  - Sample details: {details_csv_path}")


def main(output_dir, min_decreased_ratio=0.5, require_bleu_improvement=True):
    """
    Main function.

    Parameters:
    - output_dir: directory containing batch_X_summary.json files
    - min_decreased_ratio: minimum fraction of layers with distance decrease (default 0.5)
    - require_bleu_improvement: require BLEU improvement (default True)
    """
    all_samples = load_batch_results(output_dir)

    filtered_samples = filter_samples(
        all_samples,
        min_decreased_ratio=min_decreased_ratio,
        require_bleu_improvement=require_bleu_improvement,
        require_average_distance_reduction=False
    )

    if not filtered_samples:
        print("No samples meet the filtering criteria!")
        return

    layer_averages_df = calculate_layer_averages(filtered_samples)

    analyze_filtered_samples(filtered_samples)

    print("\n" + "=" * 60)
    print("LAYER-WISE AVERAGES FOR FILTERED SAMPLES")
    print("=" * 60)
    print(layer_averages_df.to_string(index=False))

    visualize_layer_averages(layer_averages_df, output_dir)

    save_filtered_results(filtered_samples, layer_averages_df, output_dir)

    return filtered_samples, layer_averages_df


if __name__ == "__main__":
    output_directory = r"<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/Results/MT/Llama2/20260116_194950/"

    filtered_samples, layer_averages = main(
        output_directory,
        min_decreased_ratio=0.3, # gpt2 0.49
        require_bleu_improvement=True
    )

    # df1 = pd.read_csv(r"D:\fine-tuning-project-local\Attention_matrix\laywer_wise_PCA_class_centroid\mt_analysis_20250724_132713\filtered_sample_details.csv")
    # df2 = pd.read_csv(r"D:\fine-tuning-project-local\Attention_matrix\laywer_wise_PCA_class_centroid\filtered_sample_details.csv")
    # # get the index of filtered samples,save to a file
    # with open("pca_distance_reduced_samples_index.txt", "w") as f:
    #     for i,row in df1.iterrows():
    #         f.write(f"{int(row[0])}\n")
    #     for i,row in df2.iterrows():
    #         # write the index of the second dataframe with an offset of 50
    #         f.write(f"{int(row[0]) + 50}\n")

    # dataset = load_dataset('kde4', lang1="en", lang2="fr")['train']
    # with open('pca_distance_reduced_samples_index.txt', 'r') as f:
    #     index_list = [int(line.strip()) for line in f]
    # dataset = dataset.select(index_list).to_csv('kde4_pca_distance_reduced_samples.csv')
