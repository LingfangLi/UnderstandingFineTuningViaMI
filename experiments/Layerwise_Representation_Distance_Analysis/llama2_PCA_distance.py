import os
import torch
import json
import gc
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformer_lens import HookedTransformer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import collections
import re
import string

# Configuration

# Options: "Sentiment", "MT", "QA"
CURRENT_TASK = "QA"

BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

FINETUNED_PATHS = {
    "Sentiment": r"<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama2-yelp",
    "MT": r"<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama2-kde4",
    "QA": r"<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama2-squad"
}

OUTPUT_DIR = f"./Results/{CURRENT_TASK}/Llama2/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 1000

# Metrics & Normalization (Official SQuAD Logic)

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

# Task Configuration

def get_task_config(task_name):
    print(f"Loading config for task: {task_name}")

    if task_name == "Sentiment":
        dataset = load_dataset('fancyzhx/yelp_polarity')['test'].select(range(NUM_SAMPLES))

        def prompt_fn(sample):
            return f"Review: {sample['text']}\nSentiment:"

        def label_fn(sample):
            return "positive" if sample['label'] == 1 else "negative"

        def metric_fn(pred, sample):
            target = label_fn(sample)
            return 1.0 if target in pred.lower() else 0.0

        return dataset, prompt_fn, label_fn, metric_fn, "accuracy"

    elif task_name == "MT":
        dataset = load_dataset('kde4', lang1="en", lang2="fr")['train'].select(range(30000, 30000+NUM_SAMPLES))

        def prompt_fn(sample):
            return (f"Translate Technical English to French.\n\n"
                    f"### Technical English:\n{sample['translation']['en']}\n\n"
                    f"### Technical French:\n")

        def label_fn(sample):
            return sample['translation']['fr']

        def metric_fn(pred, sample):
            ref = label_fn(sample)
            smoothing = SmoothingFunction().method1
            return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing)

        return dataset, prompt_fn, label_fn, metric_fn, "bleu"

    elif task_name == "QA":
        dataset = load_dataset('squad')['validation'].select(range(NUM_SAMPLES))

        def prompt_fn(sample):
            return (f"### Context:\n{sample['context']}\n\n"
                    f"### Question:\n{sample['question']}\n\n"
                    f"### Answer:\n")

        def label_fn(sample):
            return sample['answers']['text'][0]

        def metric_fn(pred, sample):
            # Max F1 over all gold answers
            gold_answers = sample['answers']['text']
            return max(compute_f1(pred, gold) for gold in gold_answers)

        return dataset, prompt_fn, label_fn, metric_fn, "f1"

# Phase Runner

def run_phase(model, dataset, text_extractor_fn, phase_name, do_generate=False):
    """
    Run model and extract features.
    Forces float32 conversion and cleans non-finite values.
    """
    results = []
    print(f"  [Executing] {phase_name} ({len(dataset)} samples)...")

    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    pad_id = model.tokenizer.pad_token_id

    for i, sample in enumerate(tqdm(dataset)):
        text_input = text_extractor_fn(sample)
        prediction = ""

        if do_generate:
            try:
                with torch.no_grad():
                    generated_text = model.generate(
                    text_input,
                    max_new_tokens=50,
                    do_sample=False,
                    eos_token_id=model.tokenizer.eos_token_id,
                    verbose=False
                    )


                    prediction = generated_text[len(text_input):].strip()
                    prediction = prediction.split('\n')[0].strip()
            except Exception as e:
                print(f"Generation Error sample {i}: {e}")
                prediction = ""

        # Feature extraction
        with torch.no_grad():
            tokens = model.to_tokens(text_input)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

            layers_vec = []
            for L in range(model.cfg.n_layers):
                vec = cache[f"blocks.{L}.hook_resid_post"][-1].cpu().numpy()

                # Force float32 to prevent sklearn overflow
                vec = vec.astype(np.float32)

                # Clean non-finite values
                if not np.isfinite(vec).all():
                    vec = np.nan_to_num(vec, nan=0.0, posinf=1e5, neginf=-1e5)

                layers_vec.append(vec)

            del cache

        results.append({
            "prediction": prediction,
            "vectors": layers_vec
        })

    return results

def load_tl_model(model_name, adapter_path=None):
    """Load model and convert to HookedTransformer."""
    print(f"Loading Model: {model_name} (Adapter: {adapter_path})...")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    if adapter_path:
        print("  Merging LoRA...")
        peft_model = PeftModel.from_pretrained(hf_model, adapter_path)
        hf_model = peft_model.merge_and_unload()
        del peft_model

    print("  Moving to GPU (HookedTransformer)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=torch.float16
    )

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return tl_model

# Main Execution

def main():
    dataset, prompt_fn, label_fn, metric_fn, metric_name = get_task_config(CURRENT_TASK)
    ft_path = FINETUNED_PATHS[CURRENT_TASK]

    # Step 1: Pretrained Model
    print("\n=== Step 1: Pretrained Model Processing ===")
    model_pre = load_tl_model(BASE_MODEL_NAME, None)

    res_input_pre = run_phase(model_pre, dataset, prompt_fn, "Pretrained Input", do_generate=True)

    res_label = run_phase(model_pre, dataset, label_fn, "Ground Truth Labels", do_generate=False)

    del model_pre
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Finetuned Model
    print("\n=== Step 2: Finetuned Model Processing ===")
    model_ft = load_tl_model(BASE_MODEL_NAME, ft_path)

    res_input_aft = run_phase(model_ft, dataset, prompt_fn, "Finetuned Input", do_generate=True)

    del model_ft
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Analysis & Calculation
    print("\n=== Step 3: Calculating Distances & Metrics ===")

    final_json = []
    n_layers = len(res_input_pre[0]['vectors'])

    # Train PCA on Pretrained Input Vectors
    pcas = {}
    scalers = {}

    for L in tqdm(range(n_layers), desc="Training PCA"):
        X = np.array([item['vectors'][L] for item in res_input_pre])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        scalers[L] = scaler
        pcas[L] = pca

    print("Computing per-sample statistics...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        pred_pre = res_input_pre[i]['prediction']
        pred_aft = res_input_aft[i]['prediction']

        score_pre = metric_fn(pred_pre, sample)
        score_aft = metric_fn(pred_aft, sample)
        metric_improvement = score_aft - score_pre

        layer_results = []
        for L in range(n_layers):
            vec_in_pre = res_input_pre[i]['vectors'][L]
            vec_in_aft = res_input_aft[i]['vectors'][L]
            vec_gt = res_label[i]['vectors'][L]

            def transform(v):
                return pcas[L].transform(scalers[L].transform(v.reshape(1, -1)))[0]

            pca_in_pre = transform(vec_in_pre)
            pca_in_aft = transform(vec_in_aft)
            pca_gt = transform(vec_gt)

            dist_before = np.linalg.norm(pca_in_pre - pca_gt)
            dist_after = np.linalg.norm(pca_in_aft - pca_gt)
            dist_change = dist_after - dist_before

            if dist_before > 1e-9:
                change_percent = (dist_change / dist_before) * 100
            else:
                change_percent = 0.0

            layer_results.append({
                "layer": L,
                "dist_before": float(dist_before),
                "dist_after": float(dist_after),
                "dist_change": float(dist_change),
                "change_percent": float(change_percent),
                "is_decreased": bool(dist_after < dist_before)
            })


        num_decreased = sum(1 for layer in layer_results if layer['is_decreased'])

        has_any_decrease = num_decreased > 0

        avg_dist_change = np.mean([layer['dist_change'] for layer in layer_results])

        item = {
            "sample_idx": i,
            f"{metric_name}_before": score_pre,
            f"{metric_name}_after": score_aft,
            f"{metric_name}_improvement": float(metric_improvement),

            "metric_improvement": float(metric_improvement),

            "prediction_before": pred_pre,
            "prediction_after": pred_aft,

            "num_decreased_layers": int(num_decreased),
            "has_any_decrease": bool(has_any_decrease),
            "avg_dist_change": float(avg_dist_change),

            "layer_distances": layer_results
        }

        if CURRENT_TASK == "QA":
            item['question'] = sample['question']
            item['answer'] = sample['answers']['text'][0]
            item['context'] = sample['context']
        elif CURRENT_TASK == "MT":
            item['source'] = sample['translation']['en']
            item['reference'] = sample['translation']['fr']
        elif CURRENT_TASK == "Sentiment":
            item['text'] = sample['text']
            item['label'] = sample['label']

        final_json.append(item)

    save_path = os.path.join(OUTPUT_DIR, "batch_1_summary.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print(f"\n Analysis Complete!")
    print(f"Results saved to: {save_path}")

if __name__ == "__main__":
    main()
