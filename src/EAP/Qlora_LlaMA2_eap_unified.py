import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import importlib
from pathlib import Path
from functools import partial
from typing import List, Union, Optional, Tuple, Literal

# TransformerLens Imports
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==========================================
# 1. Dynamic Import of EAP Library
# ==========================================
def import_eap_modules(task_name):
    """
    Dynamically import packages based on the task.
    Ensure you have eap_sentiment, eap_qa, eap_mt folders in your directory.
    """
    if task_name in ['yelp', 'twitter' ]:
        package_name = 'eap_sentiment'
    elif task_name in ['qa', 'squad', 'coqa']:
        package_name = 'eap_qa'
    elif task_name in ['mt', 'translation', 'kde4', 'tatoeba']:
        package_name = 'eap_mt'
    else:
        package_name = 'eap_sentiment'

    print(f"Importing EAP library from: {package_name}...")
    try:
        eap = importlib.import_module(package_name)
        graph = importlib.import_module(f"{package_name}.graph")
        evaluate = importlib.import_module(f"{package_name}.evaluate")
        attribute = importlib.import_module(f"{package_name}.attribute_mem")
        return graph.Graph, evaluate, attribute.attribute
    except ImportError as e:
        print(f"Error importing {package_name}. Please check folder name.")
        raise e


# ==========================================
# 2. Core Function: Prob Diff (Original Logic Restoration)
# ==========================================
def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths, labels: torch.Tensor, loss=False, mean=False):
    """
    The probability difference metric.
    Returns the difference in prob assigned to valid (correct) and invalid (incorrect) tokens.
    """
    final_logits = logits[:, -1, :]

    batch_results = []

    # 2. Iterate through Batch
    for i in range(len(labels)):
        label_idx = labels[i]

        # Get the correct answer Logit
        correct_logit = final_logits[i, label_idx]

        # Get the strongest distractor Logit
        # First set the correct position to negative infinity so it won't be selected when taking max
        current_logits = final_logits[i].clone()
        current_logits[label_idx] = -float('inf')

        max_incorrect_logit = torch.max(current_logits)

        # Calculate difference
        diff = correct_logit - max_incorrect_logit
        batch_results.append(diff)

    results = torch.stack(batch_results)

    # EAP typically requires loss (we want to minimize loss, i.e., maximize logit diff)
    # So if loss=True, we take the negative sign
    if loss:
        results = -results

    if mean:
        results = results.mean()

    return results


# ==========================================
# 3. Data Processing: Batch & Tokenization Check
# ==========================================
def batch_dataset(df, batch_size=2):
    """
    Convert DataFrame to a list of batches
    """
    # Ensure column names exist for compatibility with different naming conventions
    if 'target' in df.columns and 'label' not in df.columns:
        df['label'] = df['target']

    clean = df['clean'].tolist()
    corrupted = df['corrupted'].tolist()
    label = df['label'].tolist()

    # Batching
    clean_batches = [clean[i:i + batch_size] for i in range(0, len(df), batch_size)]
    corrupted_batches = [corrupted[i:i + batch_size] for i in range(0, len(df), batch_size)]

    # Labels here are still raw strings/numbers, will be converted to Tensor in the loop later
    label_batches = [label[i:i + batch_size] for i in range(0, len(df), batch_size)]

    return list(zip(clean_batches, corrupted_batches, label_batches))


def validate_dataset_tokenization(model, dataset):
    """
    Core function: Ensure Clean and Corrupted token lengths are exactly identical.
    If inconsistent, EAP will throw errors when computing gradients due to dimension mismatch.
    """
    print("Validating dataset tokenization...")
    fixed_dataset = []

    # Llama requires left padding
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    for clean_batch, corrupted_batch, label_batch in dataset:
        # 1. Tokenize
        clean_enc = model.tokenizer(clean_batch, padding=True, return_tensors='pt')
        corr_enc = model.tokenizer(corrupted_batch, padding=True, return_tensors='pt')

        clean_ids = clean_enc['input_ids']
        corr_ids = corr_enc['input_ids']

        clean_len = clean_ids.shape[1]
        corr_len = corr_ids.shape[1]

        if clean_len != corr_len:
            max_len = max(clean_len, corr_len)
            clean_enc = model.tokenizer(clean_batch, padding='max_length', max_length=max_len, truncation=True,
                                        return_tensors='pt')
            corr_enc = model.tokenizer(corrupted_batch, padding='max_length', max_length=max_len, truncation=True,
                                       return_tensors='pt')
            pass

        label_ids = []
        for l in label_batch:
            # If it's a numeric classification task (0/1), mapping is needed here
            if isinstance(l, int):
                txt = str(l)
            else:
                txt = " " + str(l).strip()  # Add space

            toks = model.tokenizer(txt, add_special_tokens=False)['input_ids']
            label_ids.append(toks[-1] if len(toks) > 0 else 0)  # Take the last token

        label_tensor = torch.tensor(label_ids).to(model.cfg.device)

        fixed_dataset.append((clean_batch, corrupted_batch, label_tensor))

    print(f"Validation passed. Processed {len(fixed_dataset)} batches.")
    return fixed_dataset


# ==========================================
# 4. Model Loading (Unified)
# ==========================================
def load_model_unified(model_type, model_path=None, base_model_name="meta-llama/Llama-2-7b-hf"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model: Type={model_type}, Path={model_path}")

    if model_type == 'pretrained':
        model = HookedTransformer.from_pretrained(base_model_name, device=device, fold_ln=False,
                                                  center_writing_weights=False, center_unembed=False)

    elif model_type == 'finetuned':
        from transformers import AutoModelForCausalLM
        hf_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="cpu",
                                                       trust_remote_code=True)
        hf_model = PeftModel.from_pretrained(hf_base, model_path).merge_and_unload()

        model = HookedTransformer.from_pretrained(base_model_name, hf_model=hf_model, device=device, fold_ln=False,
                                                  center_writing_weights=False, center_unembed=False)
        del hf_base, hf_model
        torch.cuda.empty_cache()

    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token = model.tokenizer.eos_token

    return model


# ==========================================
# 5. EAP Execution Logic
# ==========================================
def get_important_edges(model, dataset, metric, top_k, GraphClass, attribute_fn):
    # 1. Validate data
    dataset = validate_dataset_tokenization(model, dataset)

    # 2. Build graph
    g = GraphClass.from_model(model)

    # 3. Compute Baseline (optional)
    # baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
    # print(f"Baseline Metric: {baseline:.4f}")

    # 4. Run attribution
    print("   Running attribution...")
    attribute_fn(model, g, dataset, partial(metric, loss=True, mean=True))

    # 5. Apply threshold filtering
    scores = g.scores(absolute=True)
    top_k_score = scores[-top_k]
    print(f"   Threshold for top {top_k}: {top_k_score:.4f}")
    g.apply_threshold(top_k_score, absolute=True)

    edges = {edge_id: edge.score for edge_id, edge in g.edges.items() if edge.in_graph}
    return edges


# ==========================================
# 6. Main Process
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        choices=['yelp', 'qa', 'mt', 'squad', 'coqa', 'kde4', 'tatoeba'],default='yelp')
    parser.add_argument("--data_path", type=str, default='/users/sglli24/fine-tuning-project/llama_eap/corrupted_data/yelp_corrupted.csv')
    parser.add_argument("--ft_model_path", type=str, help="Finetuned model path",default="/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-yelp-qlora-20251119-154850/checkpoint-4950/")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--mode", type=str, default="single", choices=['single', 'compare'])
    parser.add_argument("--top_k", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges")
    args = parser.parse_args()

    # 1. Import libraries
    Graph, evaluate, attribute = import_eap_modules(args.task)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load CSV data
    print(f"Reading data: {args.data_path}")
    df = pd.read_csv(args.data_path)
    raw_dataset = batch_dataset(df, batch_size=args.batch_size)

    # 3. Analyze finetuned model
    edges_ft = None
    if args.ft_model_path:
        print("Analyzing FINETUNED Model...")
        model_ft = load_model_unified('finetuned', args.ft_model_path, args.base_model_name)
        edges_ft = get_important_edges(model_ft, raw_dataset, prob_diff, args.top_k, Graph, attribute)

        save_path = os.path.join(args.output_dir, f"llama2_{args.task}_finetuned_edges.csv")
        pd.DataFrame(list(edges_ft.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_ft
        torch.cuda.empty_cache()

    # 4. Analyze pretrained model
    edges_pt = None
    if args.mode == 'compare':
        print("Analyzing PRETRAINED Model...")
        model_pt = load_model_unified('pretrained', None, args.base_model_name)
        edges_pt = get_important_edges(model_pt, raw_dataset, prob_diff, args.top_k, Graph, attribute)

        save_path = os.path.join(args.output_dir, f"llama2_{args.task}_pretrained_edges.csv")
        pd.DataFrame(list(edges_pt.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_pt
        torch.cuda.empty_cache()

    # 5. Compute overlap
    if args.mode == 'compare' and edges_ft and edges_pt:
        common = set(edges_ft.keys()) & set(edges_pt.keys())
        print(f"Overlap Count: {len(common)}")
        common_list = [{'edge': e, 'score_ft': edges_ft[e], 'score_pt': edges_pt[e]} for e in common]
        pd.DataFrame(common_list).to_csv(os.path.join(args.output_dir, f"llama2_{args.task}_overlap.csv"), index=False)


if __name__ == "__main__":
    main()