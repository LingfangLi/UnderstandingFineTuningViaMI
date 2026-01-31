import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import re
import string
import collections

# Configuration
EXPERIMENT_TYPE = "simple"
USE_BASE_MODEL = False  # True = base model, False = fine-tuned adapter

# Paths
ADAPTER_PATH = f"<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/text_complexity/Llama2-SQuAD-complex/checkpoint-11880/"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"

TEST_INDEX_PATH = f"<PROJECT_ROOT>/experiments/text_complexity/matrix_analysis/squad_lexically_{EXPERIMENT_TYPE}_test_indices.txt"

# Output
OUTPUT_FILE = f"eval_results_squad_{EXPERIMENT_TYPE}_{'base' if USE_BASE_MODEL else 'ft'}.json"

# Metric calculation functions
def normalize_answer(s):
    """Normalize answer: remove punctuation, lowercase, remove articles"""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))

# Prompt generation
def generate_prompt(context, question):
    return (f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and base model
    print(f"Loading Base Model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Conditional adapter loading
    if USE_BASE_MODEL:
        print("\n[INFO] Evaluating BASE MODEL (Pre-trained). Skipping adapter loading.\n")
    else:
        print(f"\n[INFO] Evaluating FINE-TUNED MODEL. Loading Adapter from {ADAPTER_PATH}...\n")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()

    # Prepare data
    print(f"Loading indices from {TEST_INDEX_PATH}...")
    if not os.path.exists(TEST_INDEX_PATH):
        raise FileNotFoundError(f"Test index file not found: {TEST_INDEX_PATH}")

    with open(TEST_INDEX_PATH, 'r') as f:
        # Skip empty lines
        test_indices = [int(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(test_indices)} indices. Fetching samples from SQuAD 'train' split...")
    full_dataset = load_dataset("squad", split="train")
    eval_dataset = full_dataset.select(test_indices)
   

    print(f"Final Evaluation Set Size: {len(eval_dataset)}")

    # Evaluation loop
    total_f1 = 0
    total_em = 0
    results_data = []
    
    print("Starting generation...")
    for i, item in enumerate(tqdm(eval_dataset)):
        context = item['context']
        question = item['question']
        references = item['answers']['text'] 
        
        prompt = generate_prompt(context, question)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=30,  
                do_sample=False,    
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Post-processing
        pred_text = pred_text.strip().split('\n')[0]
        
        # Metrics
        cur_f1 = max(compute_f1(pred_text, ref) for ref in references)
        cur_em = max(compute_exact(pred_text, ref) for ref in references)
        
        total_f1 += cur_f1
        total_em += cur_em
        
        results_data.append({
            "question": question,
            "prediction": pred_text,
            "references": references,
            "f1": cur_f1,
            "em": cur_em
        })
        
        if i < 3:
            print(f"\n[Sample {i}]")
            try:
                print(f"Q Snippet: {question[:50]}")
            except: pass
            print(f"Pred: {pred_text}")
            print(f"Refs: {references}")
            print(f"F1: {cur_f1:.2f} | EM: {cur_em}")

    # Final statistics
    avg_f1 = total_f1 / len(eval_dataset)
    avg_em = total_em / len(eval_dataset)
    
    print("\n" + "="*30)
    print(f"RESULTS for {'Base Model' if USE_BASE_MODEL else ADAPTER_PATH.split('/')[-2]}")
    print(f"Average F1: {avg_f1:.4f} ({(avg_f1*100):.2f}%)")
    print(f"Average EM: {avg_em:.4f} ({(avg_em*100):.2f}%)")
    print("="*30)

    # Save results
    final_output = {
        "config": {
            "is_base_model": USE_BASE_MODEL,
            "adapter_path": ADAPTER_PATH if not USE_BASE_MODEL else "None",
            "test_indices": TEST_INDEX_PATH
        },
        "metrics": {"avg_f1": avg_f1, "avg_em": avg_em},
        "samples": results_data
    }
    
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()