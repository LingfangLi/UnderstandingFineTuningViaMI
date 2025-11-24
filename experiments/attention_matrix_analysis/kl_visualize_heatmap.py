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
            cmap="viridis",
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


# ==============================================================================
# 独立运行模式：从 CSV 读取数据并画图
# ==============================================================================
if __name__ == "__main__":
    # 1. 设置你的 CSV 文件所在根目录
    BASE_DIR = r"/mnt/scratch/users/sglli24/fine-tuning-project/attention_analysis_results"

    # 2. 指定你要重画的模型和任务
    TARGET_MODEL = "llama3"
    for TARGET_TASK in ["qa_squad", "qa_coqa", "sentiment_yelp", "mt_kde4", "mt_tatoeba"]:
            #TARGET_TASK = "qa_squad"

        csv_path = os.path.join(BASE_DIR, TARGET_MODEL, TARGET_TASK, "kl_divergence_heads.csv")

        try:
            print(f"Loading CSV from: {csv_path}")

            # 3. 读取 CSV
            # index_col=0 表示第一列是索引（Layer 0, Layer 1...），不是数据
            df = pd.read_csv(csv_path, index_col=0)

            # 4. 转换为 NumPy 矩阵
            # DataFrame 的 values 就是我们要的 [n_layers, n_heads] 矩阵
            head_matrix = df.values

            # 5. 重新计算 Layer-wise Mean (因为通常没必要单独存一个 layer mean 的 csv)
            # axis=1 表示沿着 Head 维度求平均
            layer_mean = head_matrix.mean(axis=1)

            print(f"Data shape: {head_matrix.shape}")
            print("Regenerating plots...")

            # 6. 调用绘图函数
            VisualizationEngine.plot_results(
                model_key=TARGET_MODEL,
                task_name=TARGET_TASK,
                head_matrix=head_matrix,
                layer_array=layer_mean,
                output_root_dir=BASE_DIR+"/figures/"
            )
            print("Done!")

        except FileNotFoundError:
            print(f"❌ Error: File not found at {csv_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()