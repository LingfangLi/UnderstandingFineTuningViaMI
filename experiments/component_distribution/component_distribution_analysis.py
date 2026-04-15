import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import re
import os


def parse_component_type(component_id):
    """
    Parse component type from EAP-format component ID.

    EAP naming conventions:
    - input: Embedding layer
    - aX.hY<q/k/v>: Attention head Y at layer X (query/key/value)
    - mX: MLP at layer X
    - output: Output layer / Logits
    """
    component_id = str(component_id).strip().lower()

    if component_id == 'input':
        return 'Embedding'

    if re.match(r'a\d+\.h\d+', component_id):
        return 'Attention'

    if re.match(r'^m\d+$', component_id):
        return 'MLP'

    if component_id in ['output', 'logits', 'lm_head']:
        return 'Logits'

    return 'Other'


def load_edges_from_csv(file_path, delimiter='auto'):
    """Load edge data from a CSV file."""
    if delimiter == 'auto':
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = ','

    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(delimiter)
            if len(parts) >= 2:
                edge_str = parts[0].strip()
                raw_score = parts[1].strip()

                # Skip header row
                if raw_score == 'score':
                    continue

                try:
                    score = float(raw_score)
                    edges.append((edge_str, score))
                except ValueError:
                    print(f"Warning: Skipped invalid line: {line}")
                    continue

    return edges


def analyze_component_distribution(edges, threshold_percentile=None, top_k=None):
    """
    Analyze component type distribution among edges.

    Parameters:
    - edges: list of tuples [(edge_string, importance_score), ...]
    - threshold_percentile: percentile threshold (e.g. 95 keeps top 5%)
    - top_k: alternatively, keep top k edges

    Returns:
    - component_counts: dict {component_type: count}
    - percentages: dict {component_type: percentage}
    - filtered_edges: edges after filtering
    """

    filtered_edges = edges
    if threshold_percentile is not None:
        scores = [abs(score) for _, score in edges]
        threshold = np.percentile(scores, threshold_percentile)
        filtered_edges = [(edge, score) for edge, score in edges
                          if abs(score) >= threshold]
    elif top_k is not None:
        sorted_edges = sorted(edges, key=lambda x: abs(x[1]), reverse=True)
        filtered_edges = sorted_edges[:top_k]

    print(f"Total edges: {len(edges)}, Filtered edges: {len(filtered_edges)}")

    component_counts = defaultdict(int)

    for edge_str, score in filtered_edges:
        if '->' in edge_str:
            source, target = edge_str.split('->')
            source = source.strip()
            target = target.strip()
        else:
            continue

        source_type = parse_component_type(source)
        target_type = parse_component_type(target)

        component_counts[source_type] += 1
        component_counts[target_type] += 1

    total = sum(component_counts.values())
    if total == 0:
        return {}, {}, filtered_edges

    percentages = {k: (v / total) * 100 for k, v in component_counts.items()}

    return dict(component_counts), percentages, filtered_edges


def plot_component_pie_chart(percentages, task_name="", model_name="", save_path=None):
    """
    Plot a pie chart of component type distribution.

    Parameters:
    - percentages: dict {component_type: percentage}
    - task_name: task name for title
    - model_name: model name for title
    - save_path: file path to save figure
    """
    if not percentages:
        print("No data to plot!")
        return

    # Merge small categories (<1%) into Other
    significant_components = {k: v for k, v in percentages.items()
                              if v >= 1.0 and k != 'Other'}
    other_percentage = sum(v for k, v in percentages.items()
                           if v < 1.0 or k == 'Other')

    if other_percentage > 0:
        significant_components['Other'] = other_percentage

    sorted_components = sorted(significant_components.items(),
                               key=lambda x: x[1], reverse=True)
    labels = [k for k, v in sorted_components]
    sizes = [v for k, v in sorted_components]

    color_map = {
        'Attention': '#D0E1F9',
        'MLP': '#A9C9FF',
        'Embedding': '#7DA7D9',
        'Logits': '#B19CD9',
        'LayerNorm': '#9370DB',
        'Other': '#6A5ACD'
    }
    colors = [color_map.get(label, '#' + ''.join([f'{np.random.randint(0, 256):02x}' for _ in range(3)]))
              for label in labels]

    threshold = 2.5

    # Only show labels/percentages for slices above threshold
    plot_labels = [label if size >= threshold else "" for label, size in zip(labels, sizes)]

    def my_autopct(pct):
        return ('%1.1f%%' % pct) if pct >= threshold else ''

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(sizes, labels=plot_labels, colors=colors,
                                      autopct=my_autopct, startangle=90,
                                      pctdistance=0.75,
                                      labeldistance=1.1,
                                      textprops={'fontsize': 11})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')

    title = f'Component Type Distribution'
    if model_name:
        title += f'\nModel: {model_name}'
    if task_name:
        title += f' | Task: {task_name}'

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')

    ax.legend(wedges, [f'{label}: {size:.1f}%' for label, size in zip(labels, sizes)],
              title="Components",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close('all')


def analyze_and_plot(file_path, model_name, task_name="",
                     threshold_percentile=95, top_k=None, save_path=None):
    """
    End-to-end analysis and plotting function.

    Parameters:
    - file_path: path to edge file
    - model_name: model name
    - task_name: task name
    - threshold_percentile: percentile threshold
    - top_k: alternatively use top-k filtering
    - save_path: file path to save figure
    """
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name} - {task_name}")
    print(f"File: {file_path}")
    print(f"{'=' * 60}\n")

    edges = load_edges_from_csv(file_path)
    print(f"Loaded {len(edges)} edges from file")

    counts, percentages, filtered_edges = analyze_component_distribution(
        edges, threshold_percentile=threshold_percentile, top_k=top_k
    )

    print("\nComponent Distribution:")
    print(f"{'Component Type':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    for comp_type in sorted(percentages.keys(), key=lambda x: percentages[x], reverse=True):
        print(f"{comp_type:<20} {counts[comp_type]:<10} {percentages[comp_type]:>6.2f}%")

    plot_component_pie_chart(percentages, task_name, model_name, save_path)

    return counts, percentages, filtered_edges


def compare_multiple_models(model_configs, save_path=None):
    """
    Compare component distributions across multiple models.

    Parameters:
    - model_configs: list of dicts, each containing:
        {'file_path': ..., 'model_name': ..., 'task_name': ...,
         'threshold_percentile': ..., 'top_k': ...}
    - save_path: file path to save figure
    """
    all_percentages = {}

    for config in model_configs:
        file_path = config['file_path']
        model_name = config['model_name']
        task_name = config.get('task_name', '')
        threshold_percentile = config.get('threshold_percentile', 95)
        top_k = config.get('top_k', None)

        edges = load_edges_from_csv(file_path)
        counts, percentages, _ = analyze_component_distribution(
            edges, threshold_percentile=threshold_percentile, top_k=top_k
        )

        label = f"{model_name}"
        if task_name:
            label += f" ({task_name})"
        all_percentages[label] = percentages

    fig, ax = plt.subplots(figsize=(14, 8))

    all_components = set()
    for percentages in all_percentages.values():
        all_components.update(percentages.keys())
    all_components = sorted(all_components)

    model_labels = list(all_percentages.keys())
    x = np.arange(len(model_labels))
    width = 0.8 / len(all_components) if len(all_components) > 0 else 0.8

    color_map = {
        'Attention': '#FF6B6B',
        'MLP': '#4ECDC4',
        'Embedding': '#95E1D3',
        'Logits': '#F38181',
        'LayerNorm': '#FFD93D',
        'Other': '#CCCCCC'
    }

    for i, component in enumerate(all_components):
        values = [all_percentages[model].get(component, 0) for model in model_labels]
        color = color_map.get(component, f'C{i}')
        ax.bar(x + i * width - (len(all_components) - 1) * width / 2,
               values, width, label=component, color=color)

    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Type Distribution Across Models',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha='right')
    ax.legend(title='Component Type', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    #for model_name, model_path_name in zip(["LlaMA-2-7B", "GPT2-Small", "LlaMA-3.2-1B"],["llama2", "gpt2", "llama3.2"]):
    #    for task_name in ["yelp","squad","coqa","kde4","tatoeba"]:
    #        analyze_and_plot(
     #           file_path=f"<PROJECT_ROOT>/output/EAP_edges/{model_path_name}_{task_name}_finetuned_edges.csv",
     #           model_name=model_name,
     #           task_name= task_name,
     #           threshold_percentile=0,
     #           save_path=f"./figure/{model_path_name}_{task_name}_top_400_edges_component_distribution.png"
     #       )

    # Regenerate llama2 component distribution from new full FT EAP edges (in finetuned/)
    # Other models' figures in figure/ are from earlier runs and not touched here.
    PROJECT_ROOT = "/users/sglli24/UnderstandingFineTuningViaMI"
    FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figure")
    os.makedirs(FIGURE_DIR, exist_ok=True)

    for model_name, model_path_name in [("LlaMA-2-7B", "llama2")]:
        for task_name in ["yelp", "squad", "coqa", "kde4", "tatoeba", "sst2"]:
            analyze_and_plot(
                file_path=f"{PROJECT_ROOT}/output/EAP_edges/finetuned/{model_path_name}_{task_name}_finetuned_edges.csv",
                model_name=model_name,
                task_name=task_name,
                threshold_percentile=0,
                save_path=os.path.join(FIGURE_DIR, f"{model_path_name}_{task_name}_top_400_edges_component_distribution.pdf"),
            )

    print("\n" + "=" * 80)
    print("Analyzing all models...")
    print("=" * 80)

    model_configs = [
        {
            'file_path': '<PROJECT_ROOT>/fine-tuning-project/Edges/gpt2_yelp.csv',
            'model_name': 'GPT-2 Small',
            'task_name': 'Yelp',
            'threshold_percentile': 95
        },
        {
            'file_path': '<PROJECT_ROOT>/fine-tuning-project/Edges/llama_coqa_edges.csv',
            'model_name': 'Llama-3.2-1B',
            'task_name': 'CoQA',
            'threshold_percentile': 95
        },
        {
            'file_path': '<PROJECT_ROOT>/fine-tuning-project/Edges/llama2-7b_qlora_edges.csv',
            'model_name': 'Llama2-7B',
            'task_name': 'Yelp',
            'threshold_percentile': 95
        }
    ]

    # compare_multiple_models(
    #    model_configs,
    #    save_path="/mnt/user-data/outputs/model_comparison.png"
    # )
