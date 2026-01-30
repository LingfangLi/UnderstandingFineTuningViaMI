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


def filter_samples(all_samples, min_decreased_ratio=0.5, require_accuracy_improvement=True,
                   require_average_distance_reduction=False):
    """
    Filter samples meeting specified criteria.

    Parameters:
    - min_decreased_ratio: minimum fraction of layers with decreased distance (default 0.5)
    - require_accuracy_improvement: whether accuracy must improve
    - require_average_distance_reduction: whether average distance must decrease
    """
    filtered_samples = []

    for sample in all_samples:
        accuracy_improved = sample['accuracy_after'] > sample['accuracy_before']

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
            sample['avg_dist_change'] = avg_dist_change
            distance_reduced = False

        meets_layer_criteria = decreased_ratio >= min_decreased_ratio
        meets_accuracy_criteria = accuracy_improved if require_accuracy_improvement else True
        meets_distance_criteria = distance_reduced if require_average_distance_reduction else True

        if meets_layer_criteria and meets_accuracy_criteria and meets_distance_criteria:
            sample['decreased_layers'] = decreased_layers
            sample['decreased_ratio'] = decreased_ratio
            filtered_samples.append(sample)

    print(f"\nFiltering results:")
    print(
        f"  - Samples with >{min_decreased_ratio * 100}% layers decreased: {len([s for s in all_samples if sum(1 for l in s['layer_distances'] if l['is_decreased']) / len(s['layer_distances']) >= min_decreased_ratio])}")
    print(f"  - Samples with accuracy improvement: {len([s for s in all_samples if s['accuracy_after'] > s['accuracy_before']])}")
    print(
        f"  - Samples with average distance reduction: {len([s for s in all_samples if np.mean([l['dist_after'] for l in s['layer_distances']]) < np.mean([l['dist_before'] for l in s['layer_distances']])])}")
    print(f"  - Samples meeting all selected criteria: {len(filtered_samples)}")

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

    accuracy_before = [s['accuracy_before'] for s in filtered_samples]
    accuracy_after = [s['accuracy_after'] for s in filtered_samples]
    accuracy_improvements = [s['accuracy_improvement'] for s in filtered_samples]

    print(f"\nAccuracy Statistics:")
    print(f"  Average accuracy before: {np.mean(accuracy_before):.4f}")
    print(f"  Average accuracy after: {np.mean(accuracy_after):.4f}")
    print(f"  Average accuracy improvement: {np.mean(accuracy_improvements):.4f}")
    print(f"  Max accuracy improvement: {np.max(accuracy_improvements):.4f}")
    print(f"  Min accuracy improvement: {np.min(accuracy_improvements):.4f}")

    decreased_ratios = [s['decreased_ratio'] for s in filtered_samples]
    print(f"\nLayer Decrease Statistics:")
    print(f"  Average ratio of layers with decrease: {np.mean(decreased_ratios):.2%}")
    print(f"  Samples with ALL layers decreased: {sum(1 for r in decreased_ratios if r == 1.0)}")
    print(f"  Samples with >75% layers decreased: {sum(1 for r in decreased_ratios if r > 0.75)}")

    avg_dist_changes = [s['avg_dist_change'] for s in filtered_samples]
    print(f"\nAverage Distance Change Statistics:")
    print(f"  Mean of average distance changes: {np.mean(avg_dist_changes):.4f}")
    print(f"  Samples with negative avg distance change: {sum(1 for d in avg_dist_changes if d < 0)}")

    print("\nTop 5 samples by accuracy improvement:")
    sorted_samples = sorted(filtered_samples, key=lambda x: x['accuracy_improvement'], reverse=True)
    for i, sample in enumerate(sorted_samples[:5]):
        print(f"\n  Sample {i + 1} (idx: {sample['sample_idx']}):")
        print(f"    Text: {sample.get('text', sample.get('sentence', 'N/A'))[:80]}...")
        print(f"    True label: {sample.get('label', 'N/A')}")
        print(f"    Predicted before: {sample.get('prediction_before', 'N/A')}")
        print(f"    Predicted after: {sample.get('prediction_after', 'N/A')}")
        print(f"    Accuracy: {sample['accuracy_before']:.4f} -> {sample['accuracy_after']:.4f} (+{sample['accuracy_improvement']:.4f})")
        print(f"    Layers decreased: {sample['decreased_layers']}/{len(sample['layer_distances'])}")
        print(f"    Avg distance change: {sample['avg_dist_change']:.4f}")


def visualize_layer_averages(layer_averages_df, output_dir):
    """Plot layer-wise analysis for filtered samples."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Layer-wise Analysis for Filtered Sentiment Classification Samples', fontsize=16)

    ax1 = axes[0, 0]
    x = layer_averages_df['layer']
    ax1.plot(x, layer_averages_df['avg_dist_before'], 'b-', label='Before fine-tuning', linewidth=2)
    ax1.plot(x, layer_averages_df['avg_dist_after'], 'r-', label='After fine-tuning', linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Distance')
    ax1.set_title('Average Text-Label Distance by Layer (Filtered Samples)')
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
    ax3.bar(x, layer_averages_df['decreased_percentage'], color='green', alpha=0.7)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Percentage of Samples (%)')
    ax3.set_title('Percentage of Samples with Distance Decrease by Layer')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.bar(x, layer_averages_df['avg_dist_change'],
            color=['red' if x < 0 else 'blue' for x in layer_averages_df['avg_dist_change']])
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Average Distance Change')
    ax4.set_title('Average Absolute Distance Change by Layer')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtered_layer_analysis_sentiment.png'), dpi=300)
    plt.close()


def save_filtered_results(filtered_samples, layer_averages_df, output_dir):
    """Save filtered results to JSON, CSV, and summary files."""
    filtered_json_path = os.path.join(output_dir, 'filtered_samples_sentiment.json')
    with open(filtered_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_samples, f, ensure_ascii=False, indent=2)

    layer_avg_csv_path = os.path.join(output_dir, 'filtered_layer_averages_sentiment.csv')
    layer_averages_df.to_csv(layer_avg_csv_path, index=False)

    sample_details = []
    for sample in filtered_samples:
        detail = {
            'sample_idx': sample['sample_idx'],
            'text': sample.get('text', sample.get('sentence', 'N/A')),
            'label': sample.get('label', 'N/A'),
            'prediction_before': sample.get('prediction_before', 'N/A'),
            'prediction_after': sample.get('prediction_after', 'N/A'),
            'accuracy_before': sample['accuracy_before'],
            'accuracy_after': sample['accuracy_after'],
            'accuracy_improvement': sample['accuracy_improvement'],
            'decreased_layers': sample['decreased_layers'],
            'decreased_ratio': sample['decreased_ratio'],
            'avg_dist_change': sample['avg_dist_change']
        }

        sample_details.append(detail)

    details_df = pd.DataFrame(sample_details)
    details_csv_path = os.path.join(output_dir, 'filtered_sample_details_sentiment.csv')
    details_df.to_csv(details_csv_path, index=False, encoding='utf-8')

    summary = {
        'total_filtered_samples': len(filtered_samples),
        'average_accuracy_improvement': np.mean([s['accuracy_improvement'] for s in filtered_samples]),
        'average_decreased_ratio': np.mean([s['decreased_ratio'] for s in filtered_samples]),
        'samples_with_all_layers_decreased': sum(1 for s in filtered_samples if s['decreased_ratio'] == 1.0),
        'average_distance_change': np.mean([s['avg_dist_change'] for s in filtered_samples])
    }

    summary_path = os.path.join(output_dir, 'filtered_summary_sentiment.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - Filtered samples: {filtered_json_path}")
    print(f"  - Layer averages: {layer_avg_csv_path}")
    print(f"  - Sample details: {details_csv_path}")
    print(f"  - Summary statistics: {summary_path}")


def main(output_dir, min_decreased_ratio=0.5, require_accuracy_improvement=True, require_average_distance_reduction=False):
    """
    Main function.

    Parameters:
    - output_dir: directory containing batch_X_summary.json files
    - min_decreased_ratio: minimum fraction of layers with distance decrease (default 0.5)
    - require_accuracy_improvement: require accuracy improvement (default True)
    - require_average_distance_reduction: require average distance decrease (default False)
    """
    all_samples = load_batch_results(output_dir)

    if not all_samples:
        print("No samples found!")
        return None, None

    filtered_samples = filter_samples(
        all_samples,
        min_decreased_ratio=min_decreased_ratio,
        require_accuracy_improvement=require_accuracy_improvement,
        require_average_distance_reduction=require_average_distance_reduction
    )

    if not filtered_samples:
        print("No samples meet the filtering criteria!")
        print("\nSample statistics:")
        print(f"  Total samples: {len(all_samples)}")
        if all_samples:
            accuracy_improvements = [s['accuracy_after'] - s['accuracy_before'] for s in all_samples]
            print(f"  Samples with positive accuracy improvement: {sum(1 for a in accuracy_improvements if a > 0)}")
            print(f"  Average accuracy improvement: {np.mean(accuracy_improvements):.4f}")
        return None, None

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
    output_directory = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/Results/Sentiment/Yelp/llama3/20260116_140525/"

    filtered_samples, layer_averages = main(
        output_directory,
        min_decreased_ratio=0.12,  # llama3: 0.12, gpt2 0.33
        require_accuracy_improvement=True,
        require_average_distance_reduction=False
    )

    # Alternative combinations:
    # filtered_samples, layer_averages = main(
    #     output_directory,
    #     min_decreased_ratio=0.7,
    #     require_accuracy_improvement=False,
    #     require_average_distance_reduction=True
    # )
