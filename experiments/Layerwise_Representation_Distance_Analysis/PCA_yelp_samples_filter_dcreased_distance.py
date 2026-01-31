import json
import os
import pickle
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from tqdm import tqdm
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from peft import PeftModel
from transformers import AutoModelForCausalLM

class PCAAnalysisSentiment:
    def __init__(self, pretrained_model, finetuned_model, device, n_components=2):
        self.pretrained_model = pretrained_model
        self.finetuned_model = finetuned_model
        self.device = device
        self.n_components = n_components
        self.pca_models = {}
        self.scalers = {}

    def generate_prediction(self, model, prompt, max_length=10):
        """Generate model prediction for a given prompt."""
        model.eval()
        with torch.no_grad():
            generated_text = model.generate(prompt,
                                            max_new_tokens=max_length,
                                            top_k=50,
                                            temperature=1.2)

            prediction = generated_text.replace(prompt, "").strip()

            return prediction

    def calculate_accuracy_score(self, true_label, prediction):
        """Calculate accuracy score (0 or 1) by checking if label appears in prediction."""
        true_label_str = str(true_label)

        if true_label_str in prediction:
            return 1.0
        else:
            return 0.0

    def extract_layer_representation(self, model, text, layer_idx):
        """Extract representation at a specified layer using mean pooling."""
        if not isinstance(text, str):
            raise ValueError(f"Expected text to be string, got {type(text)}")

        if not hasattr(model, 'to_tokens'):
            raise ValueError(f"Expected model to be HookedTransformer, got {type(model)}")

        tokens = model.to_tokens(text, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

        if f"blocks.{layer_idx}.hook_resid_post" in cache:
            layer_output = cache[f"blocks.{layer_idx}.hook_resid_post"]
        else:
            layer_output = cache[f"blocks.{layer_idx}.ln2.hook_normalized"]

        sentence_repr = layer_output.mean(dim=0).cpu().numpy()
        return sentence_repr

    def fit_pca_for_all_layers(self, sample_texts, num_samples=1000):
        """Fit PCA models for each layer using pretrained model representations."""
        print(f"Fitting PCA models for all layers using {num_samples} samples...")

        n_layers = self.pretrained_model.cfg.n_layers

        layer_representations = {i: [] for i in range(n_layers)}

        for text in tqdm(sample_texts[:num_samples], desc="Collecting representations"):
            for layer_idx in range(n_layers):
                repr_vec = self.extract_layer_representation(
                    self.pretrained_model, text, layer_idx
                )
                layer_representations[layer_idx].append(repr_vec)

        for layer_idx in range(n_layers):
            print(f"Fitting PCA for layer {layer_idx}...")

            X = np.array(layer_representations[layer_idx])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=self.n_components)
            pca.fit(X_scaled)

            self.scalers[layer_idx] = scaler
            self.pca_models[layer_idx] = pca

            print(f"  Layer {layer_idx}: Explained variance ratio = "
                  f"{sum(pca.explained_variance_ratio_):.4f}")

    def apply_pca_to_representation(self, repr_vec, layer_idx):
        """Reduce representation to n_components dimensions via PCA."""
        repr_scaled = self.scalers[layer_idx].transform([repr_vec])
        repr_pca = self.pca_models[layer_idx].transform(repr_scaled)[0]
        return repr_pca

    def save_batch_results(self, batch_results, batch_num, output_dir):
        """Save batch results as pickle and JSON."""
        os.makedirs(output_dir, exist_ok=True)

        pickle_path = os.path.join(output_dir, f'batch_{batch_num}_full.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(batch_results, f)

        json_data = []
        for result in batch_results:
            json_item = {
                'sample_idx': result['sample_idx'],
                'text': result['text'][:100] + '...' if len(result['text']) > 100 else result['text'],
                'label': int(result['label']),
                'prediction_before': result['prediction_before'],
                'prediction_after': result['prediction_after'],
                'accuracy_before': float(result['accuracy_before']),
                'accuracy_after': float(result['accuracy_after']),
                'accuracy_improvement': float(result['accuracy_after'] - result['accuracy_before']),
                'num_decreased_layers': int(result['num_decreased_layers']),
                'has_any_decrease': result['has_any_decrease'],
                'avg_dist_change': float(result['avg_dist_change']),
                'layer_distances': []
            }

            for layer_result in result['layer_results']:
                json_item['layer_distances'].append({
                    'layer': int(layer_result['layer']),
                    'dist_before': float(layer_result['dist_before']),
                    'dist_after': float(layer_result['dist_after']),
                    'dist_change': float(layer_result['dist_change']),
                    'change_percent': float(layer_result['change_percent']),
                    'is_decreased': bool(layer_result['is_decreased'])
                })

            json_data.append(json_item)

        json_path = os.path.join(output_dir, f'batch_{batch_num}_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        stats = {
            'batch_num': batch_num,
            'num_samples': len(batch_results),
            'avg_accuracy_before': np.mean([r['accuracy_before'] for r in batch_results]),
            'avg_accuracy_after': np.mean([r['accuracy_after'] for r in batch_results]),
            'avg_accuracy_improvement': np.mean([r['accuracy_after'] - r['accuracy_before'] for r in batch_results]),
            'samples_with_accuracy_improvement': sum(
                1 for r in batch_results if r['accuracy_after'] > r['accuracy_before']),
            'samples_with_distance_decrease': sum(1 for r in batch_results if r['has_any_decrease'])
        }

        stats_path = os.path.join(output_dir, f'batch_{batch_num}_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nBatch {batch_num} saved to {output_dir}")
        print(f"  - Average accuracy improvement: {stats['avg_accuracy_improvement']:.4f}")
        print(
            f"  - Samples with accuracy improvement: {stats['samples_with_accuracy_improvement']}/{stats['num_samples']}")
        print(f"  - Samples with distance decrease: {stats['samples_with_distance_decrease']}/{stats['num_samples']}")

    def analyze_sentiment_task_with_sample_tracking(self, test_data, num_samples=30, batch_size=1000,custom_output_dir=None):
        """Sentiment task: compute text-label representation distance with accuracy scoring."""
        if custom_output_dir:
            output_dir = custom_output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./sentiment_analysis/{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        print(f"Analyzing Sentiment Task with Sample Tracking - {num_samples} samples")
        print(f"Batch size: {batch_size}")
        print(f"Output directory: {output_dir}")

        samples_processed = []

        if isinstance(test_data, pd.DataFrame):
            for idx in range(min(num_samples, len(test_data))):
                row = test_data.iloc[idx]
                samples_processed.append({
                    'text': row['text'],
                    'label': row['label']
                })
        else:
            for i in range(min(num_samples, len(test_data))):
                sample = test_data[i]
                samples_processed.append({
                    'text': sample['text'],
                    'label': sample['label']
                })

        if not samples_processed:
            print("No valid samples found.")
            return None

        print(f"Processing {len(samples_processed)} valid samples")

        label_texts = {
            0: "negative",
            1: "positive"
        }

        batch_results = []
        batch_num = 1

        for sample_idx, sample in enumerate(tqdm(samples_processed, desc="Processing samples")):
            sample_result = {
                'sample_idx': sample_idx,
                'text': sample['text'],
                'label': sample['label'],
                'layer_results': [],
                'total_dist_before': 0,
                'total_dist_after': 0,
                'num_decreased_layers': 0,
                'has_any_decrease': False
            }

            try:
                review_prompt = f"Review: {sample['text']}\nSentiment: "
                label_text = label_texts[sample['label']]

                if sample_idx < 5:
                    print(f"\nGenerating predictions for sample {sample_idx}...")

                sample_result['prediction_before'] = self.generate_prediction(
                    self.pretrained_model, review_prompt
                )
                sample_result['prediction_after'] = self.generate_prediction(
                    self.finetuned_model, review_prompt
                )

                sample_result['accuracy_before'] = self.calculate_accuracy_score(
                    sample['label'], sample_result['prediction_before']
                )
                sample_result['accuracy_after'] = self.calculate_accuracy_score(
                    sample['label'], sample_result['prediction_after']
                )

                if sample_idx < 5:
                    print(f"  Review: {sample['text'][:100]}...")
                    print(f"  True label: {sample['label']} ({label_text})")
                    print(f"  Prediction (before): {sample_result['prediction_before']}")
                    print(f"  Prediction (after): {sample_result['prediction_after']}")
                    print(f"  Accuracy before: {sample_result['accuracy_before']}")
                    print(f"  Accuracy after: {sample_result['accuracy_after']}")

                if sample_idx < 10:
                    examples_path = os.path.join(output_dir, 'prediction_examples.txt')
                    mode = 'w' if sample_idx == 0 else 'a'
                    with open(examples_path, mode, encoding='utf-8') as f:
                        if sample_idx == 0:
                            f.write("PREDICTION EXAMPLES\n")
                            f.write("=" * 80 + "\n\n")

                        f.write(f"Sample {sample_idx + 1}:\n")
                        f.write(f"Review: {sample['text'][:200]}...\n")
                        f.write(f"True Label: {sample['label']} ({label_text})\n")
                        f.write(f"Prediction before fine-tuning: {sample_result['prediction_before']}\n")
                        f.write(f"Prediction after fine-tuning: {sample_result['prediction_after']}\n")
                        f.write(f"Correct before: {'Yes' if sample_result['accuracy_before'] == 1 else 'No'}\n")
                        f.write(f"Correct after: {'Yes' if sample_result['accuracy_after'] == 1 else 'No'}\n")
                        f.write("-" * 80 + "\n\n")

                # Compute per-layer distances
                for layer_idx in range(self.pretrained_model.cfg.n_layers):
                    # Before fine-tuning
                    review_repr_before = self.extract_layer_representation(
                        self.pretrained_model, review_prompt, layer_idx
                    )
                    label_repr_before = self.extract_layer_representation(
                        self.pretrained_model, label_text, layer_idx
                    )

                    # After fine-tuning
                    review_repr_after = self.extract_layer_representation(
                        self.finetuned_model, review_prompt, layer_idx
                    )
                    label_repr_after = self.extract_layer_representation(
                        self.finetuned_model, label_text, layer_idx
                    )

                    # PCA reduction
                    review_pca_before = self.apply_pca_to_representation(review_repr_before, layer_idx)
                    label_pca_before = self.apply_pca_to_representation(label_repr_before, layer_idx)

                    review_pca_after = self.apply_pca_to_representation(review_repr_after, layer_idx)
                    label_pca_after = self.apply_pca_to_representation(label_repr_after, layer_idx)

                    # L2 distance in PCA space
                    dist_before_pca = np.linalg.norm(review_pca_before - label_pca_before)
                    dist_after_pca = np.linalg.norm(review_pca_after - label_pca_after)

                    layer_result = {
                        'layer': layer_idx,
                        'dist_before': dist_before_pca,
                        'dist_after': dist_after_pca,
                        'dist_change': dist_after_pca - dist_before_pca,
                        'change_percent': ((
                                                       dist_after_pca - dist_before_pca) / dist_before_pca) * 100 if dist_before_pca > 0 else 0,
                        'is_decreased': dist_after_pca < dist_before_pca
                    }

                    sample_result['layer_results'].append(layer_result)
                    sample_result['total_dist_before'] += dist_before_pca
                    sample_result['total_dist_after'] += dist_after_pca

                    if layer_result['is_decreased']:
                        sample_result['num_decreased_layers'] += 1
                        sample_result['has_any_decrease'] = True

                n_layers = self.pretrained_model.cfg.n_layers
                sample_result['avg_dist_before'] = sample_result['total_dist_before'] / n_layers
                sample_result['avg_dist_after'] = sample_result['total_dist_after'] / n_layers
                sample_result['avg_dist_change'] = (sample_result['total_dist_after'] -
                                                    sample_result['total_dist_before']) / n_layers

                batch_results.append(sample_result)

                if len(batch_results) >= batch_size or sample_idx == len(samples_processed) - 1:
                    self.save_batch_results(batch_results, batch_num, output_dir)
                    batch_num += 1
                    batch_results = []

            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                continue

        self._generate_final_report(output_dir)

        return output_dir

    def _generate_final_report(self, output_dir):
        """Generate final summary report across all batches."""
        all_stats = []
        batch_num = 1

        while True:
            stats_path = os.path.join(output_dir, f'batch_{batch_num}_stats.json')

            if not os.path.exists(stats_path):
                break

            with open(stats_path, 'r') as f:
                all_stats.append(json.load(f))

            batch_num += 1

        if not all_stats:
            print("No batch statistics found.")
            return

        total_samples = sum(s['num_samples'] for s in all_stats)
        total_accuracy_improvement = sum(s['samples_with_accuracy_improvement'] for s in all_stats)
        total_distance_decrease = sum(s['samples_with_distance_decrease'] for s in all_stats)

        weighted_accuracy_before = sum(s['avg_accuracy_before'] * s['num_samples'] for s in all_stats) / total_samples
        weighted_accuracy_after = sum(s['avg_accuracy_after'] * s['num_samples'] for s in all_stats) / total_samples

        final_report = {
            'total_samples_analyzed': total_samples,
            'total_batches': len(all_stats),
            'overall_statistics': {
                'avg_accuracy_before': weighted_accuracy_before,
                'avg_accuracy_after': weighted_accuracy_after,
                'avg_accuracy_improvement': weighted_accuracy_after - weighted_accuracy_before,
                'samples_with_accuracy_improvement': total_accuracy_improvement,
                'percentage_accuracy_improvement': (total_accuracy_improvement / total_samples) * 100,
                'samples_with_distance_decrease': total_distance_decrease,
                'percentage_distance_decrease': (total_distance_decrease / total_samples) * 100
            },
            'batch_statistics': all_stats
        }

        report_path = os.path.join(output_dir, 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        summary_data = {
            'Metric': [
                'Total Samples',
                'Average Accuracy Before',
                'Average Accuracy After',
                'Average Accuracy Improvement',
                'Samples with Accuracy Improvement',
                'Percentage with Accuracy Improvement',
                'Samples with Distance Decrease',
                'Percentage with Distance Decrease'
            ],
            'Value': [
                total_samples,
                f"{weighted_accuracy_before:.4f}",
                f"{weighted_accuracy_after:.4f}",
                f"{weighted_accuracy_after - weighted_accuracy_before:.4f}",
                total_accuracy_improvement,
                f"{(total_accuracy_improvement / total_samples) * 100:.2f}%",
                total_distance_decrease,
                f"{(total_distance_decrease / total_samples) * 100:.2f}%"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, 'summary_metrics.csv')
        summary_df.to_csv(summary_csv_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"FINAL REPORT - {output_dir}")
        print(f"{'=' * 60}")
        print(f"Total samples analyzed: {total_samples}")
        print(f"Average accuracy:")
        print(f"  - Before fine-tuning: {weighted_accuracy_before:.4f}")
        print(f"  - After fine-tuning: {weighted_accuracy_after:.4f}")
        print(f"  - Improvement: {weighted_accuracy_after - weighted_accuracy_before:.4f}")
        print(
            f"Samples with accuracy improvement: {total_accuracy_improvement} ({(total_accuracy_improvement / total_samples) * 100:.1f}%)")
        print(
            f"Samples with distance decrease: {total_distance_decrease} ({(total_distance_decrease / total_samples) * 100:.1f}%)")
        print(f"\nAll results have been saved to: {output_dir}")
        print(f"Key files:")
        print(f"  - prediction_examples.txt: Sample predictions and accuracy")
        print(f"  - summary_metrics.csv: Summary statistics")
        print(f"  - final_report.json: Detailed final report")
        print(f"{'=' * 60}")


def load_model(pretrained_model, fined_model_path):
    """
    Smart model loader:
    1. Directory -> LoRA Adapter (merge into base)
    2. .pt file -> Full state dict
    """
    print(f"Attempting to load finetuned model from: {fined_model_path}")

    # LoRA Adapter directory
    if os.path.isdir(fined_model_path):
        print("Detected PEFT/LoRA adapter directory. Merging adapter into base model...")

        base_model_name = pretrained_model.cfg.model_name

        hf_base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )

        peft_model = PeftModel.from_pretrained(hf_base_model, fined_model_path)

        merged_model = peft_model.merge_and_unload()

        finetuned_model = HookedTransformer.from_pretrained(
            base_model_name,
            hf_model=merged_model,
            device=pretrained_model.cfg.device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )

        del hf_base_model, peft_model, merged_model
        torch.cuda.empty_cache()

        return finetuned_model

    # Full State Dict (.pt file)
    else:
        print("Detected full state_dict file (.pt). Loading directly...")
        cg = pretrained_model.cfg.to_dict()
        finetuned_model = HookedTransformer(cg)

        state_dict = torch.load(fined_model_path, map_location=finetuned_model.cfg.device)

        # Handle "module." prefix compatibility
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        finetuned_model.load_state_dict(new_state_dict, strict=False)
        finetuned_model.to(finetuned_model.cfg.device)
        return finetuned_model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    current_model = "qwen2"  # Options: "gpt2", "llama3", "qwen2", "llama2"

    model_configs = {
        "gpt2": {
            "hf_name": "gpt2-small",
            "ft_path": r"D:\fine-tuning\model\gpt2\sentiment\yelp.pt",
            "folder_name": "gpt2"
        },
        "llama3": {
            "hf_name": "meta-llama/Llama-3.2-1B",
            "ft_path": r"<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama3.2-yelp.pt",
            "folder_name": "llama3"
        },
        "qwen2": {
            "hf_name": "Qwen/Qwen2-0.5B",
            "ft_path": r"<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/qwen2_yelp.pt",
            "folder_name": "qwen2"
        },
        "llama2": {
            "hf_name": "meta-llama/Llama-2-7b-hf",
            "ft_path": r"<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama2-yelp",
            "folder_name": "llama2"
        }
    }

    cfg = model_configs[current_model]

    print(f"Loading pretrained model: {cfg['hf_name']}...")
    pretrained_model = HookedTransformer.from_pretrained(cfg['hf_name'], device=device)
    pretrained_model.eval()

    print("Loading datasets...")

    # Option 1: Load CSV file
    # df = pd.read_csv(r"D:\fine-tuning-project-local\Sentiment\data\yelp_lexically_complex_test.csv")
    # dataset = df

    # Option 2: HuggingFace datasets
    dataset = load_dataset('fancyzhx/yelp_polarity')['test'].select(range(1000))

    print(f"Loading finetuned model from: {cfg['ft_path']}...")
    finetuned_model = load_model(pretrained_model, cfg['ft_path'])

    analyzer = PCAAnalysisSentiment(pretrained_model, finetuned_model, device)

    print("Training PCA models...")
    training_texts = []

    if isinstance(dataset, pd.DataFrame):
        for idx in range(min(1000, len(dataset))):
            row = dataset.iloc[idx]
            prompt = f"Review: {row['text']}\nSentiment: "
            training_texts.append(prompt)
            training_texts.append("positive" if row['label'] == 1 else "negative")
    else:
        for i in range(min(1000, len(dataset))):
            sample = dataset[i]
            prompt = f"Review: {sample['text']}\nSentiment: "
            training_texts.append(prompt)
            training_texts.append("positive" if sample['label'] == 1 else "negative")

    analyzer.fit_pca_for_all_layers(training_texts, num_samples=len(training_texts))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("Results", "Sentiment", "Yelp", cfg['folder_name'], timestamp)
    print(f"Output will be saved to: {base_output_dir}")

    output_dir = analyzer.analyze_sentiment_task_with_sample_tracking(
        dataset,
        num_samples=1000,
        custom_output_dir=base_output_dir
    )
    print(f"Analysis completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
