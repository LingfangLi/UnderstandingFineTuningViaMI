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
import eap
from eap.graph import Graph
from eap import evaluate
from eap import attribute_mem as attribute
# TransformerLens Imports
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Probability difference metric
def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths, labels: torch.Tensor, loss=False, mean=False):
    """
    The probability difference metric.
    Returns the difference in prob assigned to valid (correct) and invalid (incorrect) tokens.
    """
    final_logits = logits[:, -1, :]

    batch_results = []

    for i in range(len(labels)):
        label_idx = labels[i]

        correct_logit = final_logits[i, label_idx]

        current_logits = final_logits[i].clone()
        current_logits[label_idx] = -float('inf')

        max_incorrect_logit = torch.max(current_logits)

        diff = correct_logit - max_incorrect_logit
        batch_results.append(diff)

    results = torch.stack(batch_results)

    # Negate for loss minimization (maximizes logit difference)
    if loss:
        results = -results

    if mean:
        results = results.mean()

    return results


# Dataset batching
def batch_dataset(df, batch_size=2):
    """Splits a DataFrame into batched (clean, corrupted, label) tuples."""
    if 'target' in df.columns and 'label' not in df.columns:
        df['label'] = df['target']

    clean = df['clean'].tolist()
    corrupted = df['corrupted'].tolist()
    label = df['label'].tolist()

    clean_batches = [clean[i:i + batch_size] for i in range(0, len(df), batch_size)]
    corrupted_batches = [corrupted[i:i + batch_size] for i in range(0, len(df), batch_size)]

    label_batches = [label[i:i + batch_size] for i in range(0, len(df), batch_size)]

    return list(zip(clean_batches, corrupted_batches, label_batches))


def validate_dataset_tokenization(model, dataset, task):
    """Truncates and re-encodes inputs to a safe token length to prevent OOM."""
    print("Validating dataset tokenization...")
    fixed_dataset = []

    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    SAFE_MAX_LEN = 128
    print(f"Pre-processing: Truncating text to max {SAFE_MAX_LEN} tokens to prevent OOM.")
    
    dropped_count = 0
    
    for clean_batch, corrupted_batch, label_batch in dataset:
        clean_enc = model.tokenizer(
            clean_batch, 
            truncation=True, 
            max_length=SAFE_MAX_LEN, 
            return_tensors='pt'
        )
        corr_enc = model.tokenizer(
            corrupted_batch, 
            truncation=True, 
            max_length=SAFE_MAX_LEN, 
            return_tensors='pt'
        )
        clean_ids = clean_enc['input_ids']
        corr_ids = corr_enc['input_ids']

        clean_strs = model.tokenizer.batch_decode(clean_enc['input_ids'], skip_special_tokens=True)
        corrupted_strs = model.tokenizer.batch_decode(corr_enc['input_ids'], skip_special_tokens=True)
     
        label_ids = []
        for l in label_batch:
            if task == "yelp":
                l_int = int(l)
                txt = " positive" if l_int == 1 else " negative"
            else:
                txt = " " + str(l).strip()
                
            toks = model.tokenizer(txt, add_special_tokens=False)['input_ids']
            label_ids.append(toks[-1] if len(toks) > 0 else 0)

        label_tensor = torch.tensor(label_ids).to(model.cfg.device)

        #fixed_dataset.append((clean_batch, corrupted_batch, label_tensor))  ####fixed_dataset.append((clean_strs, corrupted_strs, label_tensor))
        
        clean_strs = model.tokenizer.batch_decode(clean_ids, skip_special_tokens=True)
        corrupted_strs = model.tokenizer.batch_decode(corr_ids, skip_special_tokens=True)
        fixed_dataset.append((clean_strs, corrupted_strs, label_tensor))
        
    print(f"Validation passed. Processed {len(fixed_dataset)} batches.")
    return fixed_dataset


def load_model_unified(model_path: str, base_model_name: str = "meta-llama/Llama-2-7b-hf"):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {base_model_name} | dtype: {dtype}")

    model = None
    hf_model = None

    # Load pretrained base model
    if model_path is None or model_path.lower() == 'pretrained':
        print("  Loading pretrained base model")
        hf_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="cpu", 
            trust_remote_code=True
        )

    # Load from directory (LoRA or full fine-tuned)
    elif os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("  LoRA adapter detected, merging weights")
            hf_base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="cpu",
                trust_remote_code=True
            )
            hf_model = PeftModel.from_pretrained(hf_base, model_path)
            hf_model = hf_model.merge_and_unload()
        else:
            print("  Loading full fine-tuned model from directory")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="cpu",
                trust_remote_code=True
            )
            
    # Load from single weight file (.pt/.pth)
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        print(f"  Loading state dict from: {model_path}")

        state_dict = torch.load(model_path, map_location="cpu")

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        sample_key = next(iter(state_dict.keys()))
        is_tl_format = any("blocks." in k for k in list(state_dict.keys())[:20])
        
        if is_tl_format:
            print(f"  Detected TransformerLens format (e.g., '{sample_key}')")

            hf_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="cpu",
                trust_remote_code=True
            )

            print("  Converting base model to TransformerLens structure...")
            model = HookedTransformer.from_pretrained(
                base_model_name,
                hf_model=hf_model,
                device=device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                dtype=dtype
            )

            print("  Loading fine-tuned weights...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Load report - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            del hf_model
            hf_model = None

        else:
            print(f"  Detected HuggingFace format (e.g., '{sample_key}')")

            hf_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="cpu",
                trust_remote_code=True
            )

            missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
            print(f"  Load report - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            if len(missing) > 20:
                 raise ValueError(f"Too many missing keys ({len(missing)}). Format mismatch likely.")

    else:
        raise ValueError(f"Unknown model path format: {model_path}")

    # Convert HuggingFace model to TransformerLens
    if model is None and hf_model is not None:

        if hasattr(hf_model, "gradient_checkpointing_enable"):
            print("  Enabling gradient checkpointing")
            hf_model.config.use_cache = False 
            hf_model.gradient_checkpointing_enable()
        
        print("  Converting to TransformerLens and moving to GPU...")
        torch.cuda.empty_cache()
        
        model = HookedTransformer.from_pretrained(
            base_model_name,
            hf_model=hf_model,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=dtype
        )
        
        del hf_model

    # Final model configuration
    if model is not None:
        model.cfg.use_attn_in = True
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = False
        model.cfg.use_hook_mlp_in = True
        
        if hasattr(model.cfg, "use_cache"):
            model.cfg.use_cache = False
            
    return model

# EAP execution
import gc

def get_important_edges(model, dataset, metric, top_k, GraphClass, attribute_fn, task):
    dataset = validate_dataset_tokenization(model, dataset,task)

    torch.cuda.empty_cache()
    gc.collect()

    g = GraphClass.from_model(model)

    # baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
    # print(f"Baseline Metric: {baseline:.4f}")

    print("   Running attribution...")
    attribute_fn(model, g, dataset, partial(metric, loss=True, mean=True))

    scores = g.scores(absolute=True)
    top_k_score = scores[-top_k]
    print(f"   Threshold for top {top_k}: {top_k_score:.4f}")
    g.apply_threshold(top_k_score, absolute=True)

    edges = {edge_id: edge.score for edge_id, edge in g.edges.items() if edge.in_graph}
    
    del g
    torch.cuda.empty_cache()
    
    return edges


# Main entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,choices=['yelp', 'sst2','coqa','squad', 'kde4', 'tatoeba'],default='sst2')
    parser.add_argument("--model_name", type=str, default="llama2")  
    parser.add_argument("--output_dir", type=str, default="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/old-version-finetuned/")
    
    # Selcet mode
    parser.add_argument("--mode", type=str, default="finetuned", choices=['finetuned', 'pretrained', 'compare'])
                      
    parser.add_argument("--data_path", type=str, default='/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data/sst2_corrupted.csv')
    parser.add_argument("--ft_model_path", type=str,help="Path to fientuned model directory",default="/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/llama2-sst2-fix/") #default=/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/ "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/qwen2-0.5b-coqa-full-20251125-182058/checkpoint-4500/"
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf",help="HF Hub name for config/tokenizer") #"meta-llama/Llama-2-7b-hf" default="Qwen/Qwen2-0.5B"  "meta-llama/Llama-3.2-1B"
    parser.add_argument("--top_k", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    
    
    # Compare mode
    parser.add_argument("--edge_file_1", type=str, default=None, help="Path to first edge csv (e.g., finetuned)")
    parser.add_argument("--edge_file_2", type=str, default=None, help="Path to second edge csv (e.g., pretrained)")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading data: {args.data_path}")
    df = pd.read_csv(args.data_path)
    raw_dataset = batch_dataset(df, batch_size=args.batch_size)

    edges_ft = None
    if args.ft_model_path and args.mode == 'finetuned':
        print("Analyzing FINETUNED Model...")
        model_ft = load_model_unified(args.ft_model_path, args.base_model_name)
        edges_ft = get_important_edges(model_ft, raw_dataset, prob_diff, args.top_k, Graph, attribute.attribute,args.task)

        save_path = os.path.join(args.output_dir, f"{args.model_name}_{args.task}_finetuned_edges.csv")
        pd.DataFrame(list(edges_ft.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_ft
        torch.cuda.empty_cache()

    edges_pt = None
    if args.mode == 'pretrained':
        print("Analyzing PRETRAINED Model...")
        model_pt = load_model_unified(None, args.base_model_name)
        edges_pt = get_important_edges(model_pt, raw_dataset, prob_diff, args.top_k, Graph, attribute.attribute,args.task)
        
        pretrain_dir = os.path.join(args.output_dir, "pretrained")
        os.makedirs(pretrain_dir, exist_ok=True)
        save_path = os.path.join(pretrain_dir, f"{args.model_name}_{args.task}_pretrained_edges.csv")
        pd.DataFrame(list(edges_pt.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_pt
        torch.cuda.empty_cache()

    # Calculate overlap
    if args.mode == 'compare':
        if not args.edge_file_1 or not args.edge_file_2:
            raise ValueError("In 'compare' mode, --edge_file_1 and --edge_file_2 are required.")
        
        print(f"Comparing:\n  File 1: {args.edge_file_1}\n  File 2: {args.edge_file_2}")
        
        df1 = pd.read_csv(args.edge_file_1)
        df2 = pd.read_csv(args.edge_file_2)
        edges_1 = set(df1['edge'].tolist())
        edges_2 = set(df2['edge'].tolist())
        common = edges_1 & edges_2
        print(f"  Intersection (Common): {len(common)}")
        
        scores_1 = dict(zip(df1['edge'], df1['score']))
        scores_2 = dict(zip(df2['edge'], df2['score']))
        
        common_list = [{'edge': e, 'score_1': scores_1[e], 'score_2': scores_2[e]} for e in common]
        
        overlap_dir = os.path.join(args.output_dir, "overlap")
        os.makedirs(overlap_dir, exist_ok=True)
        
        # --- Dynamic Filename Generation Logic ---
        # Extract filename without extension (e.g., "gpt2_coqa_finetuned_edges")
        name1 = os.path.splitext(os.path.basename(args.edge_file_1))[0]
        name2 = os.path.splitext(os.path.basename(args.edge_file_2))[0]
        
        # Create combined name: "name1&name2_overlap.csv"
        # Example: gpt2_coqa_finetuned_edges&llama2_squad_finetuned_edges_overlap.csv
        save_filename = f"{name1}&{name2}_overlap.csv"
        save_path = os.path.join(overlap_dir, save_filename)
        
        pd.DataFrame(common_list).to_csv(save_path, index=False)
        print(f"Saved overlap details to: {save_path}")

if __name__ == "__main__":
    main()