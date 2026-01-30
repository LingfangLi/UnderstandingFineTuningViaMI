import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configuration
SUBSET_TYPE = "simple" # "simple" or "complex"
USE_BASE_MODEL = False # True = base model, False = fine-tuned adapter

# Paths
ADAPTER_PATH = f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/text_complexity/Llama2-7b-tatoeba-complex/checkpoint-2375/"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"

TEST_INDEX_PATH = f"/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/tatoeba_lexically_{SUBSET_TYPE}_test_indices.txt"

# Output
OUTPUT_FILE = f"eval_results_tatoeba_{SUBSET_TYPE}_{'base' if USE_BASE_MODEL else 'ft'}_nltk.json"

def generate_prompt(en_text):
    # Match training prompt format
    return f"Translate English to French. English: {en_text}\nFrench:"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and base model
    print("Loading Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Conditional adapter loading
    if USE_BASE_MODEL:
        print("\n[INFO] Evaluating BASE MODEL (Pre-trained). Skipping adapter loading.\n")
        model = base_model
    else:
        print(f"\n[INFO] Evaluating FINE-TUNED MODEL. Loading Adapter from {ADAPTER_PATH}...\n")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    model.eval()

    # Prepare data from index file
    print(f"Loading Test Indices from {TEST_INDEX_PATH}...")
    if not os.path.exists(TEST_INDEX_PATH):
        raise FileNotFoundError(f"Test index file not found: {TEST_INDEX_PATH}")
        
    with open(TEST_INDEX_PATH, 'r') as f:
        # Skip empty lines
        test_indices = [int(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(test_indices)} indices. Fetching samples from Tatoeba...")
    
    # Tatoeba only has 'train' split on HuggingFace
    full_dataset = load_dataset("tatoeba", lang1="en", lang2="fr", split="train", trust_remote_code=True)
    
    # Select the specific subset based on indices
    test_dataset = full_dataset.select(test_indices)

    
    print(f"Final Evaluation Set Size: {len(test_dataset)}")

    # Evaluation loop
    smoothing = SmoothingFunction().method1
    total_score = 0
    count = 0
    results_data = []

    print("Starting Evaluation...")
    print(f"Mode: {'BASE MODEL' if USE_BASE_MODEL else 'FINE-TUNED MODEL'}")
    
    # Batch processing
    batch_size = 8
    
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i : i + batch_size]
        en_texts = [item['en'] for item in batch['translation']]
        fr_refs = [item['fr'] for item in batch['translation']]
        
        prompts = [generate_prompt(t) for t in en_texts]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64,
                do_sample=False, # Greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_len = inputs.input_ids.shape[1]
        decoded_preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        
        for en, ref_str, pred_str in zip(en_texts, fr_refs, decoded_preds):
            # Text cleaning
            pred_clean = pred_str.strip().split('\n')[0]
            
            # Tokenize using .split() for consistency
            ref_tokens = ref_str.lower().split()
            hyp_tokens = pred_clean.lower().split()
            
            # Calculate BLEU
            if len(hyp_tokens) > 0:
                score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            else:
                score = 0.0
            
            total_score += score
            count += 1
            
            results_data.append({
                "english": en,
                "reference": ref_str,
                "prediction": pred_clean,
                "bleu": score
            })

    avg_bleu = total_score / count if count > 0 else 0
    
    print("\n" + "="*30)
    print(f"EVAL RESULTS: {SUBSET_TYPE} (NLTK .split)")
    print(f"Mode: {'Base Pre-trained' if USE_BASE_MODEL else 'Fine-tuned'}")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print("="*30)

    # Save
    final_output = {
        "config": {
            "is_base_model": USE_BASE_MODEL,
            "adapter_path": ADAPTER_PATH if not USE_BASE_MODEL else "None",
            "test_indices": TEST_INDEX_PATH
        },
        "metrics": {"avg_bleu": avg_bleu},
        "samples": results_data[:50]
    }
    
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()