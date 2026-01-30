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

class PCAAnalysisQA:
    def __init__(self, pretrained_model, finetuned_model, device, n_components=2):
        self.pretrained_model = pretrained_model
        self.finetuned_model = finetuned_model
        self.device = device
        self.n_components = n_components
        self.pca_models = {}
        self.scalers = {}

    def generate_prediction(self, model, prompt, max_length=100):
        """Generate model prediction for a given prompt."""
        model.eval()
        with torch.no_grad():
            generated_text = model.generate(prompt,
                                            max_new_tokens=max_length,
                                            top_k=50,
                                            temperature=1.0)

            prediction = generated_text.replace(prompt, "").strip()

            return prediction

    def calculate_f1_score(self, reference, hypothesis):
        """Calculate token-level F1 score."""
        reference = reference.strip().lower()
        hypothesis = hypothesis.strip().lower()

        if reference == hypothesis:
            return 1.0

        gen_tokens = set(hypothesis.split())
        ref_tokens = set(reference.split())

        common_tokens = gen_tokens.intersection(ref_tokens)

        if len(gen_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        precision = len(common_tokens) / len(gen_tokens)
        recall = len(common_tokens) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

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
                'context': result['context'][:100] + '...' if len(result['context']) > 100 else result['context'],
                'question': result['question'],
                'answer': result['answer'],
                'prediction_before': result['prediction_before'],
                'prediction_after': result['prediction_after'],
                'f1_before': float(result['f1_before']),
                'f1_after': float(result['f1_after']),
                'f1_improvement': float(result['f1_after'] - result['f1_before']),
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
            'avg_f1_before': np.mean([r['f1_before'] for r in batch_results]),
            'avg_f1_after': np.mean([r['f1_after'] for r in batch_results]),
            'avg_f1_improvement': np.mean([r['f1_after'] - r['f1_before'] for r in batch_results]),
            'samples_with_f1_improvement': sum(1 for r in batch_results if r['f1_after'] > r['f1_before']),
            'samples_with_distance_decrease': sum(1 for r in batch_results if r['has_any_decrease'])
        }

        stats_path = os.path.join(output_dir, f'batch_{batch_num}_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nBatch {batch_num} saved to {output_dir}")
        print(f"  - Average F1 improvement: {stats['avg_f1_improvement']:.4f}")
        print(f"  - Samples with F1 improvement: {stats['samples_with_f1_improvement']}/{stats['num_samples']}")
        print(f"  - Samples with distance decrease: {stats['samples_with_distance_decrease']}/{stats['num_samples']}")

    def analyze_qa_task_with_sample_tracking(self, test_data, num_samples=30, batch_size=1000, custom_output_dir=None):
        """QA task: compute question-answer representation distance with F1 scoring."""
        if custom_output_dir:
            output_dir = custom_output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./qa_analysis/{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        print(f"Analyzing QA Task with Sample Tracking - {num_samples} samples")
        print(f"Batch size: {batch_size}")
        print(f"Output directory: {output_dir}")

        samples_processed = []
        try:
            for i in range(min(num_samples, len(test_data))):
                sample = test_data[i]
                if isinstance(sample, dict) and all(key in sample for key in ['context', 'question', 'answers']):
                    answer = sample['answers']['text'][0] if sample['answers']['text'] else ""
                    samples_processed.append({
                        'context': sample['context'],
                        'question': sample['question'],
                        'answer': answer
                    })
        except Exception as e:
            print(f"Error accessing data: {e}")
            return None

        if not samples_processed:
            print("No valid samples found.")
            return None

        print(f"Processing {len(samples_processed)} valid samples")

        batch_results = []
        batch_num = 1

        for sample_idx, sample in enumerate(tqdm(samples_processed, desc="Processing samples")):
            sample_result = {
                'sample_idx': sample_idx,
                'context': sample['context'],
                'question': sample['question'],
                'answer': sample['answer'],
                'layer_results': [],
                'total_dist_before': 0,
                'total_dist_after': 0,
                'num_decreased_layers': 0,
                'has_any_decrease': False
            }

            try:
                question_prompt = f"Answer the question from the Given context. Context:{sample['context']}. Question:{sample['question']}.Answer:"
                answer_text = sample['answer']

                if sample_idx < 5:
                    print(f"\nGenerating predictions for sample {sample_idx}...")

                sample_result['prediction_before'] = self.generate_prediction(
                    self.pretrained_model, question_prompt
                )
                sample_result['prediction_after'] = self.generate_prediction(
                    self.finetuned_model, question_prompt
                )

                sample_result['f1_before'] = self.calculate_f1_score(
                    answer_text, sample_result['prediction_before']
                )
                sample_result['f1_after'] = self.calculate_f1_score(
                    answer_text, sample_result['prediction_after']
                )

                if sample_idx < 5:
                    print(f"  Question: {sample['question']}")
                    print(f"  Ground truth: {answer_text}")
                    print(f"  Prediction (before): {sample_result['prediction_before']}")
                    print(f"  Prediction (after): {sample_result['prediction_after']}")
                    print(f"  F1 before: {sample_result['f1_before']:.4f}")
                    print(f"  F1 after: {sample_result['f1_after']:.4f}")
                    print(f"  F1 improvement: {sample_result['f1_after'] - sample_result['f1_before']:.4f}")

                if sample_idx < 10:
                    examples_path = os.path.join(output_dir, 'prediction_examples.txt')
                    mode = 'w' if sample_idx == 0 else 'a'
                    with open(examples_path, mode, encoding='utf-8') as f:
                        if sample_idx == 0:
                            f.write("PREDICTION EXAMPLES\n")
                            f.write("=" * 80 + "\n\n")

                        f.write(f"Sample {sample_idx + 1}:\n")
                        f.write(f"Context: {sample['context'][:200]}...\n")
                        f.write(f"Question: {sample['question']}\n")
                        f.write(f"Ground Truth Answer: {answer_text}\n")
                        f.write(f"Prediction before fine-tuning: {sample_result['prediction_before']}\n")
                        f.write(f"Prediction after fine-tuning: {sample_result['prediction_after']}\n")
                        f.write(f"F1 before: {sample_result['f1_before']:.4f}\n")
                        f.write(f"F1 after: {sample_result['f1_after']:.4f}\n")
                        f.write(f"F1 improvement: {sample_result['f1_after'] - sample_result['f1_before']:.4f}\n")
                        f.write("-" * 80 + "\n\n")

                # Compute per-layer distances
                for layer_idx in range(self.pretrained_model.cfg.n_layers):
                    # Before fine-tuning
                    question_repr_before = self.extract_layer_representation(
                        self.pretrained_model, question_prompt, layer_idx
                    )
                    answer_repr_before = self.extract_layer_representation(
                        self.pretrained_model, answer_text, layer_idx
                    )

                    # After fine-tuning
                    question_repr_after = self.extract_layer_representation(
                        self.finetuned_model, question_prompt, layer_idx
                    )
                    answer_repr_after = self.extract_layer_representation(
                        self.finetuned_model, answer_text, layer_idx
                    )

                    # PCA reduction
                    question_pca_before = self.apply_pca_to_representation(question_repr_before, layer_idx)
                    answer_pca_before = self.apply_pca_to_representation(answer_repr_before, layer_idx)

                    question_pca_after = self.apply_pca_to_representation(question_repr_after, layer_idx)
                    answer_pca_after = self.apply_pca_to_representation(answer_repr_after, layer_idx)

                    # L2 distance in PCA space
                    dist_before_pca = np.linalg.norm(question_pca_before - answer_pca_before)
                    dist_after_pca = np.linalg.norm(question_pca_after - answer_pca_after)

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
        total_f1_improvement = sum(s['samples_with_f1_improvement'] for s in all_stats)
        total_distance_decrease = sum(s['samples_with_distance_decrease'] for s in all_stats)

        weighted_f1_before = sum(s['avg_f1_before'] * s['num_samples'] for s in all_stats) / total_samples
        weighted_f1_after = sum(s['avg_f1_after'] * s['num_samples'] for s in all_stats) / total_samples

        final_report = {
            'total_samples_analyzed': total_samples,
            'total_batches': len(all_stats),
            'overall_statistics': {
                'avg_f1_before': weighted_f1_before,
                'avg_f1_after': weighted_f1_after,
                'avg_f1_improvement': weighted_f1_after - weighted_f1_before,
                'samples_with_f1_improvement': total_f1_improvement,
                'percentage_f1_improvement': (total_f1_improvement / total_samples) * 100,
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
                'Average F1 Before',
                'Average F1 After',
                'Average F1 Improvement',
                'Samples with F1 Improvement',
                'Percentage with F1 Improvement',
                'Samples with Distance Decrease',
                'Percentage with Distance Decrease'
            ],
            'Value': [
                total_samples,
                f"{weighted_f1_before:.4f}",
                f"{weighted_f1_after:.4f}",
                f"{weighted_f1_after - weighted_f1_before:.4f}",
                total_f1_improvement,
                f"{(total_f1_improvement / total_samples) * 100:.2f}%",
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
        print(f"Average F1 score:")
        print(f"  - Before fine-tuning: {weighted_f1_before:.4f}")
        print(f"  - After fine-tuning: {weighted_f1_after:.4f}")
        print(f"  - Improvement: {weighted_f1_after - weighted_f1_before:.4f}")
        print(
            f"Samples with F1 improvement: {total_f1_improvement} ({(total_f1_improvement / total_samples) * 100:.1f}%)")
        print(
            f"Samples with distance decrease: {total_distance_decrease} ({(total_distance_decrease / total_samples) * 100:.1f}%)")
        print(f"\nAll results have been saved to: {output_dir}")
        print(f"Key files:")
        print(f"  - prediction_examples.txt: Sample predictions and F1 scores")
        print(f"  - summary_metrics.csv: Summary statistics")
        print(f"  - final_report.json: Detailed final report")
        print(f"{'=' * 60}")


def load_model(pretrained_model, fined_model_path):
    print(f"Attempting to load finetuned model from: {fined_model_path}")

    # LoRA adapter directory
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

    # .pt state dict file
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

    # Configuration
    current_model_key = "qwen2"  # Options: "gpt2", "llama2-7b", "qwen2"
    dataset_name = 'squad'  # Options: 'squad' or 'coqa'

    base_path = r"/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models"

    model_configs = {
        "gpt2": {
            "hf_name": "gpt2-small",
            "squad_path": os.path.join(base_path, "gpt2-squad.pt"),
            "coqa_path": os.path.join(base_path, "gpt2-coqa.pt"),
            "folder_name": "GPT2_Small"
        },
        "llama2": {
            "hf_name": "meta-llama/Llama-2-7b-hf",
            "squad_path": os.path.join(base_path, "llama2-squad"),
            "coqa_path": os.path.join(base_path, "llama2-coqa"),
            "folder_name": "Llama2_7B"
        },
        "llama3": {
            "hf_name": "meta-llama/Llama-3.2-1B",
            "squad_path": os.path.join(base_path, "llama3.2-squad.pt"),
            "coqa_path": os.path.join(base_path, "llama3.2-coqa.pt"),
            "folder_name": "llama3"
        },
        "qwen2": {
            "hf_name": "Qwen/Qwen2-0.5B",
            "squad_path": os.path.join(base_path, "qwen2_squad.pt"),
            "coqa_path": os.path.join(base_path, "qwen2_coqa.pt"),
            "folder_name": "Qwen2_0.5B"
        }
    }

    cfg = model_configs[current_model_key]
    ft_path = cfg['squad_path'] if dataset_name == 'squad' else cfg['coqa_path']

    print(f"=== Running QA Experiment: {cfg['folder_name']} on {dataset_name.upper()} ===")

    print("Loading pretrained model {cfg['hf_name']}...")
    pretrained_model = HookedTransformer.from_pretrained(cfg['hf_name'], device=device)
    pretrained_model.eval()

    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'squad':
        dataset = load_dataset('squad')['validation'].select(range(1000))
    else:
        dataset = load_dataset('coqa')['validation'].select(range(500))

    print(f"Loading finetuned model from: {ft_path}...")
    finetuned_model = load_model(pretrained_model, ft_path)

    analyzer = PCAAnalysisQA(pretrained_model, finetuned_model, device)

    print("Training PCA models...")
    print("Dataset info:")
    print(f"  - Length: {len(dataset)}")
    print(f"  - Type: {type(dataset)}")
    if len(dataset) > 0:
        print(f"  - Sample keys: {dataset[0].keys()}")

    training_texts = []
    limit = 1000 if len(dataset) > 1000 else len(dataset)
    for i in range(limit):
        sample = dataset[i]
        if dataset_name == 'squad':
            context = sample['context']
            question = sample['question']
            answer = sample['answers']['text'][0] if sample['answers']['text'] else ""
        else: # coqa
            context = sample['story']
            question = sample['questions'][0]
            answer = sample['answers']['input_text'][0]

        prompt = f"Answer the question from the Given context. Context:{context}. Question:{question}.Answer:"
        training_texts.append(prompt)
        training_texts.append(answer)
    analyzer.fit_pca_for_all_layers(training_texts, num_samples=len(training_texts))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    structured_output_dir = os.path.join(
        "Model_Internal_States_Analysis",
        "Results",
        "QA",
        dataset_name.upper(),
        cfg['folder_name'],
        timestamp
    )
    print(f"Results will be saved to: {structured_output_dir}")

    output_dir = analyzer.analyze_qa_task_with_sample_tracking(
        dataset,
        num_samples=1000,
        custom_output_dir=structured_output_dir
    )
    print(f"Analysis completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
