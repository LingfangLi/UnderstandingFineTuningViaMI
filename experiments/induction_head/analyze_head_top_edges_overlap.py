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

# CSV Loader: Parse EAP results
class EAPLoader:
    @staticmethod
    def load_from_csv(csv_path, n_layers, n_heads):
        """Parse head importance matrix from EAP CSV file."""
        if not os.path.exists(csv_path):
            logging.error(f"CSV not found: {csv_path}")
            return np.zeros((n_layers, n_heads))

        logging.info(f"Loading EAP scores from: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Failed to read CSV: {e}")
            return np.zeros((n_layers, n_heads))

        importance_matrix = np.zeros((n_layers, n_heads))

        # Match "a{number}.h{number}" pattern
        pattern = re.compile(r"a(\d+)\.h(\d+)")

        count = 0
        for _, row in df.iterrows():
            edge_name = str(row['edge'])
            score = float(row['score'])
            abs_score = abs(score)

            matches = pattern.findall(edge_name)

            for (layer_idx, head_idx) in matches:
                l, h = int(layer_idx), int(head_idx)

                if l < n_layers and h < n_heads:
                    importance_matrix[l, h] += abs_score
                    count += 1

        logging.info(f"   Processed {count} edges/nodes related to attention heads.")

        if np.max(importance_matrix) > 0:
            importance_matrix = importance_matrix / np.max(importance_matrix)

        return importance_matrix

# Analyzer
class OverlapAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

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

    def get_important_heads(self, importance_matrix, top_k=50):
        n_layers, n_heads = importance_matrix.shape
        flat = []
        for l in range(n_layers):
            for h in range(n_heads):
                flat.append( ((l,h), importance_matrix[l,h]) )

        flat.sort(key=lambda x: x[1], reverse=True)

        selected_set = set()
        for i in range(min(top_k, len(flat))):
            selected_set.add(flat[i][0])

        return selected_set

    def analyze(self, model_key, task_name, is_ft, importance_matrix):
        save_dir = os.path.join(self.output_dir, model_key)

        ind_set, ind_scores = self.load_induction_heads(model_key, task_name, is_ft)

        if ind_scores is None: return

        k_target = max(len(ind_set), 20)

        imp_set = self.get_important_heads(importance_matrix, top_k=800)

        overlap_set = ind_set.intersection(imp_set)
        overlap_count = len(overlap_set)

        logging.info(f"[Result {model_key}-{task_name}]")
        logging.info(f"   Induction Heads: {len(ind_set)}")
        logging.info(f"   Important Heads (Top-{k_target}): {len(imp_set)}")
        logging.info(f"   Overlap: {overlap_count}")

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
                    hue.append("Both (CRITICAL)")
                elif coord in ind_set:
                    hue.append("Induction Only")
                elif coord in imp_set:
                    hue.append("Important Only")
                else:
                    hue.append("Other")

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x, y=y, hue=hue, palette={
            "Both (CRITICAL)": "red",
            "Induction Only": "blue",
            "Important Only": "orange",
            "Other": "lightgrey"
        }, alpha=0.6, s=50)

        plt.title(f"Correlation: {model_key} on {task_name}")
        plt.xlabel("Induction Score (Mechanism)")
        plt.ylabel("EAP Importance Score (Task Relevance)")

        path = os.path.join(save_dir, f"scatter_{task_name}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logging.info(f"Plot saved to {path}")

# Configuration and Main Program
if __name__ == "__main__":
    OUTPUT_DIR = r"<PROJECT_ROOT>/experiments/induction_head/output/"
    EAP_CSV_DIR = r"<PROJECT_ROOT>/output/EAP_edges/finetuned/"

    # Model parameters (n_layers, n_heads)
    MODEL_CONFIGS = {
        "gpt2": (12, 12),
        "llama2": (32, 32),
        "llama3": (16, 32),
        "qwen2": (24, 16)
    }

    # Task-to-CSV mapping
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

    for model_key, tasks in CSV_MAP.items():
        n_layers, n_heads = MODEL_CONFIGS[model_key]

        for task_name, csv_file in tasks.items():
            csv_full_path = os.path.join(EAP_CSV_DIR, csv_file)

            importance_matrix = EAPLoader.load_from_csv(csv_full_path, n_layers, n_heads)

            # Only analyze FineTuned version by default
            analyzer.analyze(model_key, task_name, is_ft=True, importance_matrix=importance_matrix)
