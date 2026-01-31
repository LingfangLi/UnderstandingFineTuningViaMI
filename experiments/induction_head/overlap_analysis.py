import torch
import numpy as np
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# CSV Loader: Matrix mode for plots, Set mode for stats
class EAPLoader:
    @staticmethod
    def load_matrix_from_csv(csv_path, n_layers, n_heads):
        """Parse head importance matrix from EAP CSV file for visualization."""
        if not os.path.exists(csv_path):
            logging.error(f"CSV not found: {csv_path}")
            return np.zeros((n_layers, n_heads))

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Failed to read CSV: {e}")
            return np.zeros((n_layers, n_heads))

        importance_matrix = np.zeros((n_layers, n_heads))
        pattern = re.compile(r"a(\d+)\.h(\d+)")

        for _, row in df.iterrows():
            edge_name = str(row['edge'])
            score = float(row['score'])
            abs_score = abs(score)

            matches = pattern.findall(edge_name)
            for (layer_idx, head_idx) in matches:
                l, h = int(layer_idx), int(head_idx)
                if l < n_layers and h < n_heads:
                    importance_matrix[l, h] += abs_score

        if np.max(importance_matrix) > 0:
            importance_matrix = importance_matrix / np.max(importance_matrix)

        return importance_matrix

    @staticmethod
    def get_heads_from_top_edges(csv_path, top_k_edges, n_layers, n_heads):
        """
        Extract involved head set from top-K edges.
        Returns: Set of (layer, head) tuples.
        """
        if not os.path.exists(csv_path):
            return set()

        try:
            df = pd.read_csv(csv_path)
        except:
            return set()

        df['abs_score'] = df['score'].abs()
        df = df.sort_values(by='abs_score', ascending=False)

        top_df = df.head(top_k_edges)

        important_heads_set = set()
        pattern = re.compile(r"a(\d+)\.h(\d+)")

        for edge_str in top_df['edge']:
            matches = pattern.findall(str(edge_str))

            for (layer_idx, head_idx) in matches:
                l, h = int(layer_idx), int(head_idx)
                if l < n_layers and h < n_heads:
                    important_heads_set.add((l, h))

        return important_heads_set

# Analyzer
class OverlapAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results_buffer = []

    def load_induction_heads(self, model_key, task_name, is_ft):
        status = "FineTuned" if is_ft else "Base"
        suffix = f"{task_name}" if is_ft else "Pretrained"
        json_path = os.path.join(self.output_dir, model_key, f"detected_heads_{status}_{suffix}.json")
        npy_path = os.path.join(self.output_dir, model_key, f"induction_scores_{status}_{suffix}.npy")

        if not os.path.exists(json_path):
            logging.error(f"Induction JSON not found: {json_path}")
            return set(), None

        with open(json_path, 'r') as f:
            data = json.load(f)

        induction_set = set()
        for h in data['heads']:
            induction_set.add((h['layer'], h['head']))

        induction_scores = np.load(npy_path)
        return induction_set, induction_scores

    def analyze(self, model_key, task_name, is_ft, csv_path, n_layers, n_heads, top_k_edges=400):
        save_dir = os.path.join(self.output_dir, model_key)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        ind_set, ind_scores = self.load_induction_heads(model_key, task_name, is_ft)
        if ind_scores is None: return

        # Get important heads directly from top-K edges
        imp_set = EAPLoader.get_heads_from_top_edges(csv_path, top_k_edges, n_layers, n_heads)

        # Load matrix for visualization only
        importance_matrix = EAPLoader.load_matrix_from_csv(csv_path, n_layers, n_heads)

        # Calculate Overlap Stats
        overlap_set = ind_set.intersection(imp_set)

        num_ind = len(ind_set)
        num_imp = len(imp_set)
        num_overlap = len(overlap_set)

        recall = (num_overlap / num_ind * 100) if num_ind > 0 else 0.0
        precision = (num_overlap / num_imp * 100) if num_imp > 0 else 0.0

        logging.info(f"[Result {model_key}-{task_name}] Top-{top_k_edges} Edges contain {num_imp} heads.")
        logging.info(f"   Induction: {num_ind} | Overlap: {num_overlap} | Recall: {recall:.1f}% | Precision: {precision:.1f}%")

        self.results_buffer.append({
            "Model": f"{model_key}-{task_name}",
            "Top_K_Edges_Cutoff": top_k_edges,
            "Num_Induction_Heads": num_ind,
            "Num_Important_Heads_in_Edges": num_imp,
            "Overlap_Count": num_overlap,
            "Recall (%)": round(recall, 2),
            "Precision (%)": round(precision, 2),
            "Overlap_Heads": str(sorted(list(overlap_set)))
        })

        self.visualize(ind_scores, importance_matrix, ind_set, imp_set, overlap_set, model_key, task_name, save_dir)

    def visualize(self, ind_scores, imp_scores, ind_set, imp_set, overlap_set, model_key, task_name, save_dir):
        n_layers, n_heads = ind_scores.shape
        x, y, hue = [], [], []

        for l in range(n_layers):
            for h in range(n_heads):
                x.append(ind_scores[l,h])
                y.append(imp_scores[l,h])

                coord = (l,h)
                if coord in overlap_set:
                    hue.append("Both (Overlap)")
                elif coord in ind_set:
                    hue.append("Induction Only")
                elif coord in imp_set:
                    hue.append("Important Only")
                else:
                    hue.append("Other")

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x, y=y, hue=hue, palette={
            "Both (Overlap)": "red",
            "Induction Only": "blue",
            "Important Only": "orange",
            "Other": "lightgrey"
        }, alpha=0.6, s=50)

        plt.title(f"Correlation: {model_key} on {task_name}")
        plt.xlabel("Induction Score")
        plt.ylabel("EAP Importance Score (Accumulated)")

        path = os.path.join(save_dir, f"scatter_{task_name}.pdf")
        plt.savefig(path, dpi=300)
        plt.close()

    def save_summary_table(self, filename="final_overlap_stats.csv"):
        """Save final statistics table."""
        if not self.results_buffer:
            return
        df = pd.DataFrame(self.results_buffer)
        out_path = os.path.join(self.output_dir, filename)
        df.to_csv(out_path, index=False)
        logging.info(f"Summary Table saved to: {out_path}")

# Configuration and Main Program
if __name__ == "__main__":
    OUTPUT_DIR = r"<PROJECT_ROOT>/experiments/induction_head/output/"
    EAP_CSV_DIR = r"<PROJECT_ROOT>/output/EAP_edges/finetuned/"

    MODEL_CONFIGS = {
        "gpt2": (12, 12),
        "llama2": (32, 32),
        "llama3": (16, 32),
        "qwen2": (24, 16)
    }

    CSV_MAP = {
        "gpt2": {
            "sentiment_yelp": "gpt2_yelp_finetuned_edges.csv",
            "sentiment_sst2": "gpt2_sst2_finetuned_edges.csv",
            "qa_squad":       "gpt2_squad_finetuned_edges.csv",
            "qa_coqa":        "gpt2_coqa_finetuned_edges.csv",
            "mt_kde4":        "gpt2_kde4_finetuned_edges.csv",
            "mt_tatoeba":     "gpt2_tatoeba_finetuned_edges.csv"
        },
        "llama3": {
            "sentiment_yelp": "llama3.2_yelp_finetuned_edges.csv",
            "sentiment_sst2": "llama3.2_sst2_finetuned_edges.csv",
            "qa_squad":       "llama3.2_squad_finetuned_edges.csv",
            "qa_coqa":        "llama3.2_coqa_finetuned_edges.csv",
            "mt_kde4":        "llama3.2_kde4_finetuned_edges.csv",
            "mt_tatoeba":     "llama3.2_tatoeba_finetuned_edges.csv"
        },
        "llama2": {
            "sentiment_yelp": "llama2_yelp_finetuned_edges.csv",
            "sentiment_sst2-fix": "llama2_sst2_finetuned_edges.csv",
            "qa_squad":       "llama2_squad_finetuned_edges.csv",
            "qa_coqa":        "llama2_coqa_finetuned_edges.csv",
            "mt_kde4":        "llama2_kde4_finetuned_edges.csv",
            "mt_tatoeba":     "llama2_tatoeba_finetuned_edges.csv"
        },
        "qwen2": {
            "sentiment_yelp": "qwen2_yelp_finetuned_edges.csv",
            "sentiment_sst2": "qwen2_sst2_finetuned_edges.csv",
            "qa_squad":       "qwen2_squad_finetuned_edges.csv",
            "qa_coqa":        "qwen2_coqa_finetuned_edges.csv",
            "mt_kde4":        "qwen2_kde4_finetuned_edges.csv",
            "mt_tatoeba":     "qwen2_tatoeba_finetuned_edges.csv"
        }
    }

    analyzer = OverlapAnalyzer(OUTPUT_DIR)

    TOP_K_EDGES = 400

    for model_key, tasks in CSV_MAP.items():
        n_layers, n_heads = MODEL_CONFIGS[model_key]

        for task_name, csv_file in tasks.items():
            csv_full_path = os.path.join(EAP_CSV_DIR, csv_file)

            analyzer.analyze(
                model_key=model_key,
                task_name=task_name,
                is_ft=True,
                csv_path=csv_full_path,
                n_layers=n_layers,
                n_heads=n_heads,
                top_k_edges=TOP_K_EDGES
            )

    analyzer.save_summary_table("induction_overlap_stats_edges400.csv")
