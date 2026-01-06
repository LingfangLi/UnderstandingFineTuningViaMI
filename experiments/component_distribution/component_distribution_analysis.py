import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import re


def parse_component_type(component_id):
    """
    根据EAP格式的组件ID解析组件类型

    EAP组件命名约定:
    - input: Embedding层
    - aX.hY<q/k/v>: 第X层注意力的第Y个头（query/key/value）
    - mX: 第X层的MLP
    - output: 输出层/Logits
    """
    component_id = str(component_id).strip().lower()

    # Embedding层
    if component_id == 'input':
        return 'Embedding'

    # 注意力头 (格式: aX.hY<q/k/v> 或 aX.hY)
    if re.match(r'a\d+\.h\d+', component_id):
        return 'Attention'

    # MLP层 (格式: mX)
    if re.match(r'^m\d+$', component_id):
        return 'MLP'

    # 输出层/Logits
    if component_id in ['output', 'logits', 'lm_head']:
        return 'Logits'

    # 其他情况
    return 'Other'


def load_edges_from_csv(file_path, delimiter='auto'):
    """
    从CSV文件加载边数据
    """
    # 自动检测分隔符
    if delimiter == 'auto':
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = ','

    # 读取数据
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            # 注意：删除了这里的 next(f)，因为它会跳过数据行
            
            line = line.strip()
            if not line:
                continue

            parts = line.split(delimiter)
            if len(parts) >= 2:
                edge_str = parts[0].strip()
                raw_score = parts[1].strip() # 先获取字符串

                # --- 核心修复开始 ---
                # 1. 明确判断：如果第二列是单词 'score'，说明是表头，跳过
                if raw_score == 'score':
                    continue
                
                # 2. 安全转换：使用 try-except 防止其他非数字导致崩溃
                try:
                    score = float(raw_score)
                    edges.append((edge_str, score))
                except ValueError:
                    print(f"Warning: Skipped invalid line: {line}")
                    continue

    return edges


def analyze_component_distribution(edges, threshold_percentile=None, top_k=None):
    """
    分析边中各组件的类型分布

    Parameters:
    - edges: list of tuples [(edge_string, importance_score), ...]
    - threshold_percentile: 百分位数阈值（例如95表示只保留前5%）
    - top_k: 或者直接指定保留前k条边

    Returns:
    - component_counts: dict {component_type: count}
    - percentages: dict {component_type: percentage}
    - filtered_edges: 筛选后的边列表
    """

    # 根据阈值筛选边
    filtered_edges = edges
    if threshold_percentile is not None:
        scores = [abs(score) for _, score in edges]  # 使用绝对值
        threshold = np.percentile(scores, threshold_percentile)
        filtered_edges = [(edge, score) for edge, score in edges
                          if abs(score) >= threshold]
    elif top_k is not None:
        # 按绝对值排序，取前k个
        sorted_edges = sorted(edges, key=lambda x: abs(x[1]), reverse=True)
        filtered_edges = sorted_edges[:top_k]

    print(f"Total edges: {len(edges)}, Filtered edges: {len(filtered_edges)}")

    # 统计组件类型
    component_counts = defaultdict(int)

    for edge_str, score in filtered_edges:
        # 解析边: "source->target"
        if '->' in edge_str:
            source, target = edge_str.split('->')
            source = source.strip()
            target = target.strip()
        else:
            continue

        # 解析组件类型并统计
        source_type = parse_component_type(source)
        target_type = parse_component_type(target)

        component_counts[source_type] += 1
        component_counts[target_type] += 1

    # 计算百分比
    total = sum(component_counts.values())
    if total == 0:
        return {}, {}, filtered_edges

    percentages = {k: (v / total) * 100 for k, v in component_counts.items()}

    return dict(component_counts), percentages, filtered_edges


def plot_component_pie_chart(percentages, task_name="", model_name="", save_path=None):
    """
    绘制组件类型分布的饼图

    Parameters:
    - percentages: dict {component_type: percentage}
    - task_name: 任务名称
    - model_name: 模型名称
    - save_path: 保存路径
    """
    if not percentages:
        print("No data to plot!")
        return

    # 过滤掉占比太小的类别（<1%），合并为Other
    significant_components = {k: v for k, v in percentages.items()
                              if v >= 1.0 and k != 'Other'}
    other_percentage = sum(v for k, v in percentages.items()
                           if v < 1.0 or k == 'Other')

    if other_percentage > 0:
        significant_components['Other'] = other_percentage

    # 按百分比排序
    sorted_components = sorted(significant_components.items(),
                               key=lambda x: x[1], reverse=True)
    labels = [k for k, v in sorted_components]
    sizes = [v for k, v in sorted_components]

    # 颜色配置 - 使用专业的配色方案
    color_map = {
        'Attention': '#FF6B6B',  # 红色
        'MLP': '#4ECDC4',  # 青色
        'Embedding': '#95E1D3',  # 浅绿
        'Logits': '#F38181',  # 粉红
        'LayerNorm': '#FFD93D',  # 黄色
        'Other': '#CCCCCC'  # 灰色
    }
    colors = [color_map.get(label, '#' + ''.join([f'{np.random.randint(0, 256):02x}' for _ in range(3)]))
              for label in labels]

    # 1. 定义一个阈值，比如 3%
    threshold = 2.5 

    # 2. 处理标签：只有大于阈值的才显示 Label，否则为空
    plot_labels = [label if size >= threshold else "" for label, size in zip(labels, sizes)]

    # 3. 处理百分比：只有大于阈值的才显示数值，否则为空
    def my_autopct(pct):
        return ('%1.1f%%' % pct) if pct >= threshold else ''

    # 创建饼图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 注意这里传入的是 plot_labels 而不是原始的 labels
    wedges, texts, autotexts = ax.pie(sizes, labels=plot_labels, colors=colors,
                                      autopct=my_autopct, startangle=90,
                                      pctdistance=0.75,  # 调整百分比文字距离圆心的位置
                                      labeldistance=1.1, # 调整标签文字距离圆心的位置
                                      textprops={'fontsize': 11})

    # 美化文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')

    # 标题
    title = f'Component Type Distribution'
    if model_name:
        title += f'\nModel: {model_name}'
    if task_name:
        title += f' | Task: {task_name}'

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')

    # 添加图例
    ax.legend(wedges, [f'{label}: {size:.1f}%' for label, size in zip(labels, sizes)],
              title="Components",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def analyze_and_plot(file_path, model_name, task_name="",
                     threshold_percentile=95, top_k=None, save_path=None):
    """
    一站式分析和绘图函数

    Parameters:
    - file_path: 边文件路径
    - model_name: 模型名称
    - task_name: 任务名称
    - threshold_percentile: 百分位数阈值
    - top_k: 或使用top-k筛选
    - save_path: 保存路径
    """
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name} - {task_name}")
    print(f"File: {file_path}")
    print(f"{'=' * 60}\n")

    # 加载边数据
    edges = load_edges_from_csv(file_path)
    print(f"Loaded {len(edges)} edges from file")

    # 分析组件分布
    counts, percentages, filtered_edges = analyze_component_distribution(
        edges, threshold_percentile=threshold_percentile, top_k=top_k
    )

    # 打印统计信息
    print("\nComponent Distribution:")
    print(f"{'Component Type':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    for comp_type in sorted(percentages.keys(), key=lambda x: percentages[x], reverse=True):
        print(f"{comp_type:<20} {counts[comp_type]:<10} {percentages[comp_type]:>6.2f}%")

    # 绘制饼图
    plot_component_pie_chart(percentages, task_name, model_name, save_path)

    return counts, percentages, filtered_edges


def compare_multiple_models(model_configs, save_path=None):
    """
    对比多个模型的组件分布

    Parameters:
    - model_configs: list of dicts, 每个dict包含:
        {'file_path': ..., 'model_name': ..., 'task_name': ...,
         'threshold_percentile': ..., 'top_k': ...}
    - save_path: 保存路径
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

    # 创建对比图
    fig, ax = plt.subplots(figsize=(14, 8))

    # 获取所有组件类型
    all_components = set()
    for percentages in all_percentages.values():
        all_components.update(percentages.keys())
    all_components = sorted(all_components)

    # 准备数据
    model_labels = list(all_percentages.keys())
    x = np.arange(len(model_labels))
    width = 0.8 / len(all_components) if len(all_components) > 0 else 0.8

    # 颜色映射
    color_map = {
        'Attention': '#FF6B6B',
        'MLP': '#4ECDC4',
        'Embedding': '#95E1D3',
        'Logits': '#F38181',
        'LayerNorm': '#FFD93D',
        'Other': '#CCCCCC'
    }

    # 为每种组件类型创建条形
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


# 使用示例
if __name__ == "__main__":
    #for model_name, model_path_name in zip(["LlaMA-2-7B", "GPT2-Small", "LlaMA-3.2-1B"],["llama2", "gpt2", "llama3.2"]):
    #    for task_name in ["yelp","squad","coqa","kde4","tatoeba"]:
            # 示例：分析单个模型
    #        analyze_and_plot(
     #           file_path=f"/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/{model_path_name}_{task_name}_finetuned_edges.csv",
     #           model_name=model_name,
     #           task_name= task_name,
     #           threshold_percentile=0,  # 保留前5%重要的边
     #           save_path=f"./figure/{model_path_name}_{task_name}_top_400_edges_component_distribution.png"
     #       )
    
    #Analysis single model
    for model_name, model_path_name in zip(["LlaMA-2-7B", "GPT2-Small", "LlaMA-3.2-1B"],["llama2", "gpt2", "llama3.2"]):# in zip(["Qwen2-0.5B"],["qwen2"]):
        for task_name in ["sst2"]:#["yelp","squad","coqa","kde4","tatoeba","sst2"]:
            # 示例：分析单个模型
            analyze_and_plot(
                file_path=f"/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/{model_path_name}_{task_name}_finetuned_edges.csv",
                model_name=model_name,
                task_name= task_name,
                threshold_percentile=0,  # 保留前5%重要的边
                save_path=f"./figure/{model_path_name}_{task_name}_top_400_edges_component_distribution.png"
            )        

    # 分析所有三个模型
    print("\n" + "=" * 80)
    print("Analyzing all models...")
    print("=" * 80)

    model_configs = [
        {
            'file_path': '/users/sglli24/fine-tuning-project/Edges/gpt2_yelp.csv',
            'model_name': 'GPT-2 Small',
            'task_name': 'Yelp',
            'threshold_percentile': 95
        },
        {
            'file_path': '/users/sglli24/fine-tuning-project/Edges/llama_coqa_edges.csv',
            'model_name': 'Llama-3.2-1B',
            'task_name': 'CoQA',
            'threshold_percentile': 95
        },
        {
            'file_path': '/users/sglli24/fine-tuning-project/Edges/llama2-7b_qlora_edges.csv',
            'model_name': 'Llama2-7B',
            'task_name': 'Yelp',
            'threshold_percentile': 95
        }
    ]

    # 对比分析
    # compare_multiple_models(
    #    model_configs,
    #    save_path="/mnt/user-data/outputs/model_comparison.png"
    # )