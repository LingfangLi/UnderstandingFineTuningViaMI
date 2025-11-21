import torch
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple, Literal
from functools import partial
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
# Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache

device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("WARNING: Running on CPU.")

from huggingface_hub import login

### Token for llama3.2-1b model
# login(token="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")
### Token for llama2-7b model
login(token="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")


def setup_model(model_path: str):
    """Load and configure a model for EAP analysis"""
    model1 = HookedTransformer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                               device="cuda:0" if torch.cuda.is_available() else "cpu")
    cg = model1.cfg.to_dict()
    model = HookedTransformer(cg)
    model.load_state_dict(torch.load(model_path))
    model.to(model.cfg.device)
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


def setup_pretrained_model(model_name: str):
    """Load a pretrained model from Hugging Face"""
    model = HookedTransformer.from_pretrained(model_name, device="cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(model.cfg.device)
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


def load_qlora_model():
    """Load and merge QLoRA correctly, save as standard FP16 format"""
    from peft import PeftModel, PeftConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from pathlib import Path

    peft_model_id = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-yelp-qlora-20251119-154850/checkpoint-4950"
    config = PeftConfig.from_pretrained(peft_model_id)

    # 1. Load base model (specify torch_dtype explicitly, disable quantization)
    print("Loading base model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,  # Key: Use FP16, not 4-bit
        device_map="cpu",
        trust_remote_code=False,  # Do not add load_in_4bit=True or any quantization_config
    )

    # 2. Load PEFT weights
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    # 3. Merge weights (this generates standard weights)
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # 4. Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 5. Save to a new directory (avoid contaminating old files)
    save_dir = Path("/mnt/scratch/users/sglli24/fine-tuning-models/llama2-7b-qlora-best")
    save_dir.mkdir(exist_ok=True)

    # 6. Save standard weights (now clean FP16)
    merged_state_dict_path = save_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), merged_state_dict_path)

    # Save tokenizer and config
    tokenizer.save_pretrained(save_dir)
    model.config.save_pretrained(save_dir)

    print(f"✅ Clean FP16 weights saved to {save_dir}")
    print(f" Weight file size: {merged_state_dict_path.stat().st_size / 1024 ** 3:.2f} GB")

    # 7. Thoroughly clean cache
    del base_model, model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return save_dir


from transformers import AutoModelForCausalLM
from transformer_lens.pretrained.weight_conversions import convert_llama_weights
from copy import deepcopy


def setup_merged_llama2(save_dir):
    """Load standard FP16 weights and convert to TLens format"""
    # 1. Get TLens config (keep as object form)
    print("Loading TLens config...")
    tlens_pretrained = HookedTransformer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        device="cpu"
    )
    cfg = tlens_pretrained.cfg  # Must be object, not dict

    # 2. Load HuggingFace model object (now it's clean)
    print(f"Loading HF model from {save_dir}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=False
    )

    # 3. Key: Call official conversion function
    print("Converting weights from HF to TLens format...")
    tlens_state_dict = convert_llama_weights(hf_model, cfg)

    # 4. Create TLens model and load
    model = HookedTransformer(cfg)
    missing, unexpected = model.load_state_dict(tlens_state_dict, strict=False)

    # 5. Print debug information
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print(f" Example: {missing[:3]}")
    if unexpected:
        print(f" Example: {unexpected[:3]}")

    # 6. Move to device and configure
    model.to(device)
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # 7. Verify
    print("Verifying weight conversion...")
    model.eval()
    with torch.no_grad():
        test_prompt = "The movie is great"
        test_tokens = model.tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(device)
        logits = model(test_tokens)
        if torch.isnan(logits).any():
            print("❌ ERROR: Logits contain NaN!")
        elif abs(logits.mean().item()) < 1e-3:
            print("❌ ERROR: Logits are near-zero!")
        else:
            print(f"✅ SUCCESS: Logits mean = {logits.mean().item():.4f}, std = {logits.std().item():.4f}")
            print(f" Shape: {logits.shape}")

    # 8. Clean up
    del hf_model, tlens_pretrained
    torch.cuda.empty_cache()

    return model


def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths, labels: torch.Tensor, loss=False, mean=False):
    """
    the probability difference metric, which takes in logits and labels (years), and returns the difference in prob. assigned to valid (> year) and invalid (<= year) tokens (corrupted_logits and input_lengths are due to the Graph framework introduced below)
    """
    probs = torch.softmax(logits[:, -1], dim=-1)
    results = []
    probs, next_tokens = torch.topk(probs[-1], 5)
    prob_a = 0
    prob_b = 0
    for prob, token, label in zip(probs, next_tokens, labels):
        if token == label:
            prob_b = prob
        else:
            prob_a = prob_a + prob
    results = prob_b - prob_a
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


def batch_dataset(df, batch_size=2):
    clean, corrupted, label = [df[col].tolist() for col in ['clean', 'corrupted', 'label']]
    clean = [clean[i:i + batch_size] for i in range(0, len(clean), batch_size)]
    corrupted = [corrupted[i:i + batch_size] for i in range(0, len(corrupted), batch_size)]
    label = [torch.tensor(label[i:i + batch_size]) for i in range(0, len(label), batch_size)]
    return [(clean[i], corrupted[i], label[i]) for i in range(len(clean))]


def validate_dataset_tokenization(model, dataset):
    """Validate that clean and corrupted examples have same tokenized length"""
    fixed_dataset = []
    for clean_batch, corrupted_batch, label in dataset:
        # Process each batch
        fixed_clean = []
        fixed_corrupted = []
        for clean, corrupted in zip(clean_batch, corrupted_batch):
            # Tokenize both
            clean_ids = model.tokenizer(clean, return_tensors='pt')['input_ids']
            corrupted_ids = model.tokenizer(corrupted, return_tensors='pt')['input_ids']
            clean_len = clean_ids.size(1)
            corrupted_len = corrupted_ids.size(1)
            if clean_len == corrupted_len:
                # Lengths already same, use directly
                fixed_clean.append(clean)
                fixed_corrupted.append(corrupted)
            elif 'X' in corrupted and clean_len > corrupted_len:
                # corrupted has X and is shorter than clean, need to increase number of X
                # Calculate number of tokens to supplement
                token_diff = clean_len - corrupted_len
                # Find X position and replace with multiple X
                # First try simple replacement of single X
                if corrupted.count('X') == 1:
                    # Only one X, directly replace with multiple
                    X_replacement = ' '.join(['X'] * (token_diff + 1))
                    corrupted_fixed = corrupted.replace('X', X_replacement)
                else:
                    # Multiple X, distribute evenly
                    num_x = corrupted.count('X')
                    x_per_position = (token_diff + num_x) // num_x
                    X_replacement = ' '.join(['X'] * x_per_position)
                    corrupted_fixed = corrupted.replace('X', X_replacement)
                # Verify length after fixing
                corrupted_fixed_ids = model.tokenizer(corrupted_fixed, return_tensors='pt')['input_ids']
                if corrupted_fixed_ids.size(1) == clean_len:
                    fixed_clean.append(clean)
                    fixed_corrupted.append(corrupted_fixed)
                else:
                    # If still doesn't match, use padding scheme
                    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
                    max_len = max(clean_len, corrupted_fixed_ids.size(1))
                    # Pad both to max length
                    clean_padded = model.tokenizer(clean, padding='max_length', max_length=max_len, truncation=True,
                                                   return_tensors='pt')
                    corrupted_padded = model.tokenizer(corrupted_fixed, padding='max_length', max_length=max_len,
                                                       truncation=True, return_tensors='pt')
                    fixed_clean.append(model.tokenizer.decode(clean_padded['input_ids'][0], skip_special_tokens=False))
                    fixed_corrupted.append(
                        model.tokenizer.decode(corrupted_padded['input_ids'][0], skip_special_tokens=False))
            else:
                # Other cases (no X or corrupted longer), use padding
                pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
                max_len = max(clean_len, corrupted_len)
                # Pad to same length
                if clean_len < max_len:
                    clean_ids = torch.nn.functional.pad(clean_ids, (0, max_len - clean_len), value=pad_token_id)
                if corrupted_len < max_len:
                    corrupted_ids = torch.nn.functional.pad(corrupted_ids, (0, max_len - corrupted_len),
                                                            value=pad_token_id)
                # Decode back to text
                fixed_clean.append(model.tokenizer.decode(clean_ids[0], skip_special_tokens=False))
                fixed_corrupted.append(model.tokenizer.decode(corrupted_ids[0], skip_special_tokens=False))
        fixed_dataset.append((fixed_clean, fixed_corrupted, label))
    print("Dataset tokenization validation: all pairs aligned.")
    return fixed_dataset


import eap_sentiment as eap
from eap_sentiment.graph import Graph
from eap_sentiment import evaluate
from eap_sentiment import attribute_mem as attribute


def get_important_edges(model, dataset, metric, top_k=400):
    """Run EAP and return the important edges"""
    # Validate dataset first
    dataset = validate_dataset_tokenization(model, dataset)

    # Create graph from model
    g = Graph.from_model(model)

    # Evaluate baseline
    baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
    print(f"Baseline: {baseline}")

    # Run attribution
    attribute.attribute(model, g, dataset, partial(metric, loss=True, mean=True))

    # Apply threshold to get top edges
    scores = g.scores(absolute=True)
    print("scores[-400]", scores[-top_k])
    #g.apply_threshold(scores[-top_k], absolute=True)
    # print("edge num after patching:",sum(edge.in_graph for edge in g.edges.values()))

    # Get edge information
    edges = {edge_id: {'score': edge.score, 'abs_score': abs(edge.score), 'source': str(edge.parent),
                       'target': str(edge.child)} for edge_id, edge in g.edges.items() if edge.in_graph}
    return g, edges


def compute_edge_overlap(edges1: dict, edges2: dict):
    """Compute overlap between two sets of edges"""
    # Get edge IDs from both sets
    edge_ids1 = set(edges1.keys())
    edge_ids2 = set(edges2.keys())

    # Compute intersection
    common_edges = edge_ids1.intersection(edge_ids2)

    # Overlap metrics
    overlap_count = len(common_edges)

    # Get details of common edges
    common_edge_details = []
    for edge_id in common_edges:
        common_edge_details.append({
            'edge_id': edge_id,
            'score1': edges1[edge_id]['score'],
            'score2': edges2[edge_id]['score'],
            'abs_score1': edges1[edge_id]['abs_score'],
            'abs_score2': edges2[edge_id]['abs_score'],
            'source': edges1[edge_id]['source'],
            'target': edges1[edge_id]['target']
        })

    # Sort by average absolute score
    common_edge_details.sort(key=lambda x: (x['abs_score1'] + x['abs_score2']) / 2, reverse=True)
    return {
        'overlap_count': overlap_count,
        'common_edges': common_edge_details
    }


import pandas as pd
import numpy as np
import ast
import re
import json


def text_to_dict_alternative(path):
    """Alternative method that handles Python dict format directly"""
    if path.endswith('.txt'):
        with open(path, "r") as f:
            content = f.read().strip()
        # Create a namespace with numpy
        import numpy as np
        from types import SimpleNamespace
        Node = lambda x: x  # Define Node as identity function

        # Use eval with a restricted namespace for safety
        try:
            namespace = {'np': np, 'Node': Node}
            result = eval(content, namespace)

            # Convert numpy types to Python types
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, np.float32):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            result = convert_numpy(result)
            print(f"Successfully parsed {len(result)} entries")
            return result
        except Exception as e:
            print(f"Error evaluating content: {e}")
            raise
    elif path.endswith('.csv'):
        # Change sep to ',' for comma-separated
        df = pd.read_csv(path, sep='\t', header=None, names=['edge_id', 'score'])
        edges_dict = {}
        for _, row in df.iterrows():
            edge_id = row['edge_id']
            score = row['score']
            try:
                score = float(score)
                abs_score = abs(score)

                # Extract source and target from edge_id (e.g., 'input->a0.h0<q>' -> source='input', target='a0.h0<q>')
                parts = edge_id.split('->')
                if len(parts) != 2:
                    raise ValueError(f"Invalid edge_id format: {edge_id}")

                source = parts[0]
                target = parts[1]

                edges_dict[edge_id] = {
                    'score': score,
                    'abs_score': abs_score,
                    'source': source,  # Use 'input' or 'Node(input)' as needed
                    'target': target  # Use 'a0.h0<q>' or 'Node(a0.h0)' as needed
                }
            except ValueError as e:
                print(f"Skipping row with edge_id '{edge_id}': invalid score '{score}' ({e})")
                continue

        print(f"Processed {len(edges_dict)} edges from {path}")
        return edges_dict


def main():
    try:
        df1 = pd.read_csv('/users/sglli24/fine-tuning-project/llama_eap/corrupted_data/yelp_corrupted.csv')
        print(f"Loaded dataset with {len(df1)} samples")
    except FileNotFoundError:
        print("Error: './a.csv' not found!")
        # return

    # Prepare datasets
    dataset1 = batch_dataset(df1, batch_size=2)
    print(f"Prepared {len(dataset1)} batches")

    # Load models
    try:
        ### Load the finetuned model
        # model1 = setup_model("/users/sglli24/model_500.pt") #"/mnt/data1/users/sglli24/fine-tuning-project-1/model/llama/qa/coqa_llama.pt"
        ###Load the pretrained model
        # model1= setup_pretrained_model("meta-llama/Llama-2-7b-hf")
        ### Load qlora tuned model
        save_dir = load_qlora_model()
        model1 = setup_merged_llama2(save_dir)
        print(f"Model 1 loaded on {model1.cfg.device}")
    except FileNotFoundError:
        print("Error: '../model/llama2' not found!")
        return

    # try:
    #     model2 = setup_model("final.pt")
    #     print(f"Model 2 loaded on {model2.cfg.device}")
    # except FileNotFoundError:
    #     print("Error: '../models/Yelp_v1.pt' not found!")
    #     return

    # Define metric
    metric = prob_diff

    # # Get important edges for both models
    print("Analyzing Model 1...")
    g1, edges1 = get_important_edges(model1, dataset1, metric, top_k=400)
    # print(edges1)
    ### Build a file to store top edge information
    with open("/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/llama2-7b_qlora_full_edges.csv", "w") as f:
        for key in edges1:
            f.write(key)
            f.write("\t")
            f.write(str(edges1[key]["score"]))
            f.write("\n")
        f.close()

    # print("\nAnalyzing Model 2...")
    # g2, edges2 = get_important_edges(model2, dataset1, metric, top_k=400)

    # ### Compute the edge overlap of pretrained and finetuned model version by giving 2 edges path. ### Comment out when need use
    # import re
    # import os
    # directory1 = "/users/sglli24/fine-tuning-project/Edges/llama_pretrained/"
    # directory2 = "/users/sglli24/fine-tuning-project/Edges/llama_finetuned/"
    # for item1 in os.listdir(directory1):
    #    for item2 in os.listdir(directory2):
    #        file1 = os.path.join(directory1,item1)
    #        file2 = os.path.join(directory2,item2)
    #        dataset1 = re.search(r'llama_(\w+)_edges', item1)
    #        dataset2 = re.search(r'llama_(\w+)_edges', item2)
    #
    #        if dataset1 and dataset2:
    #            dataset_name1 = dataset1.group(1)
    #            dataset_name2 = dataset2.group(1)
    #        else:
    #            parts1 = item1.replace('.csv', '').split('_')
    #            dataset_name1 = parts1[-2]
    #            parts2 = item2.replace('.csv', '').split('_')
    #            dataset_name2 = parts2[-2]
    #        edges1 = text_to_dict_alternative(file1)
    #        edges2= text_to_dict_alternative(file2)
    #
    #        # Compute overlap
    #        overlap_results = compute_edge_overlap(edges1, edges2)
    #        print(f"Processing {dataset_name1} (pretrained) vs {dataset_name2} (finetuned)")
    #        print(f"Common edges: {overlap_results['overlap_count']}")
    #
    #        # Save detailed results
    #        #pd.DataFrame(overlap_results['common_edges']).to_csv('/users/sglli24/fine-tuning-project/Edges/llama_finetined_yelp_corrupt_twitter&llama_finetuned_yelp_common_edges.csv', index=False)
    #        output_path = f"/users/sglli24/fine-tuning-project/Edges/llama_pretrained_finetuned_common_edges/llama_pretrained_{dataset_name1}_finetuned_{dataset_name2}_common_edges.csv"
    #        pd.DataFrame(overlap_results['common_edges']).to_csv(output_path, index=False)
    #        print(f"Saved to: {output_path}\n")

    # return g1, g2, overlap_results


if __name__ == "__main__":
    main()