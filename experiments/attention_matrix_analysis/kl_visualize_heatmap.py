import os
import numpy as np
import pandas as pd  # 需要导入 pandas
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VisualizationEngine:
    @staticmethod
    def plot_results(model_key: str, task_name: str, head_matrix: np.ndarray, layer_array: np.ndarray,
                     output_root_dir: str):
        """
        生成并保存可视化图表。
        """
        # 1. 准备保存路径
        save_dir = os.path.join(output_root_dir, model_key, task_name)
        os.makedirs(save_dir, exist_ok=True)

        # 设置 Seaborn 风格
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

        # =======================================================
        # 图 1: Head-wise Heatmap (热力图)
        # =======================================================
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

        heatmap_path = os.path.join(save_dir, f"{model_key}_{task_name}_heatmap_head_kl.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved Heatmap to: {heatmap_path}")

        # =======================================================
        # 图 2: Layer-wise Bar Plot (柱状图)
        # =======================================================
        plt.figure(figsize=(12, 6))

        layers = np.arange(len(layer_array))

        sns.barplot(x=layers, y=layer_array, color="#4c72b0", alpha=0.9)

        plt.title(f"Layer-wise Average Shift: {model_key} on {task_name}", fontsize=18, pad=20)
        plt.xlabel("Layer Index", fontsize=14)
        plt.ylabel("Avg KL Divergence", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # 优化 X 轴标签
        if len(layers) > 20:
            for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
                if i % 2 != 0:
                    label.set_visible(False)

        barplot_path = os.path.join(save_dir, f"{model_key}_{task_name}_barplot_layer_kl.png")
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved Barplot to: {barplot_path}")
        
    @staticmethod
    def plot_model_summary(model_key: str, task_layer_means: dict, output_root_dir: str, data_root_dir: str = None):
        """
        [新增功能] 生成模型在不同任务下的层级变化对比热力图。
        X轴: 任务 (Tasks)
        Y轴: 层 (Layers)
        颜色: 该层平均 KL 散度
        """
        if not task_layer_means:
            logging.warning(f"No data found for model {model_key}, skipping summary plot.")
            return

        # 1. 准备数据: 将字典转换为 DataFrame (行=Layers, 列=Tasks)
        # task_layer_means 结构: {'qa_squad': [0.1, 0.2...], 'sentiment_yelp': [0.5, ...]}
        df_summary = pd.DataFrame(task_layer_means)
        
        # 确保列的顺序一致性 (可选，按字母排序)
        df_summary = df_summary.reindex(sorted(df_summary.columns), axis=1)
        
        if data_root_dir:
            # 存入 /attention_analysis_results/{model_key}/
            csv_save_dir = os.path.join(data_root_dir, model_key)
            os.makedirs(csv_save_dir, exist_ok=True)
            
            csv_path = os.path.join(csv_save_dir, f"{model_key}_layer_wise_summary.csv")
            df_summary.to_csv(csv_path, index=True)
            logging.info(f"Saved Summary CSV to: {csv_path}")
            
        # 2. 准备保存路径
        save_dir = os.path.join(output_root_dir, model_key)
        os.makedirs(save_dir, exist_ok=True)

        # 3. 绘图
        plt.figure(figsize=(10, 8))  # 宽度取决于任务数量，10寸够5-6个任务了
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

        ax = sns.heatmap(
            df_summary,
            cmap="Blues",
            cbar_kws={'label': 'Avg KL Divergence (Layer-wise)'}
        )

        plt.title(f"Layer-wise Shifts across Tasks: {model_key}", fontsize=16, pad=20)
        plt.xlabel("Task", fontsize=14)
        plt.ylabel("Layer", fontsize=14)
        
        # 翻转Y轴，让Layer 0在下面
        plt.gca().invert_yaxis()
        
        # 旋转X轴标签，防止重叠
        plt.xticks(rotation=45, ha='right')

        # 4. 保存
        summary_path = os.path.join(save_dir, f"{model_key}_ALL_TASKS_layer_heatmap.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved Model Summary Heatmap to: {summary_path}")    


# ==============================================================================
# 独立运行模式：从 CSV 读取数据并画图
# ==============================================================================
if __name__ == "__main__":
    # 1. 设置你的 CSV 文件所在根目录
    BASE_DIR = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/attention_matrix_analysis/attention_analysis_results/"
    OUTPUT_ROOT_FIGS = os.path.join(BASE_DIR, "figures")

    MODELS = ["gpt2", "llama2", "llama3", "qwen2"]
    TASKS = ["qa_squad", "qa_coqa", "sentiment_yelp", "mt_kde4", "mt_tatoeba"]

    # 2. 指定你要重画的模型和任务
    for TARGET_MODEL in MODELS:
        print(f"\n{'='*40}")
        print(f"Processing Model: {TARGET_MODEL}")
        print(f"{'='*40}")
        
        # 用于收集该模型下所有任务的层级平均值
        # 结构: { "qa_squad": np.array([l0_mean, l1_mean...]), ... }
        model_tasks_data = {}

        for TARGET_TASK in TASKS:
            csv_path = os.path.join(BASE_DIR, TARGET_MODEL, TARGET_TASK, "kl_divergence_heads.csv")
            
            try:
                # 3. 读取 CSV
                if not os.path.exists(csv_path):
                    print(f"Skipping {TARGET_TASK}: File not found at {csv_path}")
                    continue

                print(f"Loading {TARGET_TASK}...")
                df = pd.read_csv(csv_path, index_col=0)
                
                # 4. 转换为 NumPy 矩阵 & 计算层平均值
                head_matrix = df.values
                layer_mean = head_matrix.mean(axis=1) # shape: (n_layers,)
                
                # 5. 收集数据用于最后的总图
                model_tasks_data[TARGET_TASK] = layer_mean

                # 6. [原有功能] 为每个任务生成独立的详细图表
                VisualizationEngine.plot_results(
                    model_key=TARGET_MODEL,
                    task_name=TARGET_TASK,
                    head_matrix=head_matrix,
                    layer_array=layer_mean,
                    output_root_dir=OUTPUT_ROOT_FIGS
                )
            
            except Exception as e:
                print(f"❌ Error processing {TARGET_TASK}: {e}")
                import traceback
                traceback.print_exc()

        # 7. [新增功能] 该模型所有任务跑完后，画一张总的热力图
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