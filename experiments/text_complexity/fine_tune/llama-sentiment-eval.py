import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import json
import os

# Configuration
SUBSET_TYPE = "simple"  # "simple" or "complex"
USE_BASE_MODEL = False  # True = base model, False = fine-tuned adapter

# Paths
ADAPTER_PATH = f"<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/text_complexity/Llama2-Yelp-complex/checkpoint-1250/"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"

TEST_INDEX_PATH = f"<PROJECT_ROOT>/experiments/text_complexity/matrix_analysis/yelp_lexically_{SUBSET_TYPE}_test_indices.txt"

# Output
OUTPUT_FILE = f"eval_results_yelp_{SUBSET_TYPE}_{'base' if USE_BASE_MODEL else 'ft'}.json"

def generate_prompt(text):
    # Clean text (consistent with training)
    clean_text = text.replace('\n', ' ').strip()
    return f"Review: {clean_text}\nSentiment:"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Left padding for generation

    # Load base model (4-bit quantization)
    print("Loading Base Model...")
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

    # Prepare test data
    print(f"Loading Test Indices from {TEST_INDEX_PATH}...")
    if not os.path.exists(TEST_INDEX_PATH):
        raise FileNotFoundError(f"Test index file not found: {TEST_INDEX_PATH}")
    
    with open(TEST_INDEX_PATH, 'r') as f:
        # Skip empty lines
        test_indices = [int(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(test_indices)} indices. Fetching samples from Yelp...")
    
    raw_test = load_dataset("yelp_polarity", split="train").select(test_indices)

    print(f"Evaluated on {len(raw_test)} samples ({SUBSET_TYPE}).")

    # Evaluation loop
    predictions = []
    ground_truths = []
    results_data = []

    print("Starting generation...")
    
    batch_size = 8
    
    for i in tqdm(range(0, len(raw_test), batch_size)):
        batch = raw_test[i : i + batch_size]
        texts = batch['text']
        labels = batch['label'] # 0: Negative, 1: Positive
        
        # Construct prompts
        prompts = [generate_prompt(t) for t in texts]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            # Generate
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10,   
                do_sample=False,     # Greedy search
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for text, pred_str, true_label in zip(texts, decoded_preds, labels):
            # Post-processing
            pred_clean = pred_str.strip().lower()
            
            # Parsing logic
            predicted_label = -1 
            
            if "positive" in pred_clean:
                predicted_label = 1
            elif "negative" in pred_clean:
                predicted_label = 0
            else:
                predicted_label = -1 
            
            predictions.append(predicted_label)
            ground_truths.append(true_label)
            
            # Save examples
            if len(results_data) < 5:
                results_data.append({
                    "text_snippet": text[:100],
                    "prediction_raw": pred_str,
                    "prediction_mapped": predicted_label,
                    "ground_truth": true_label,
                    "correct": predicted_label == true_label
                })

    # Calculate metrics
    acc = accuracy_score(ground_truths, predictions)
    
    # Check for valid predictions before generating classification report
    if any(p != -1 for p in predictions):
        report = classification_report(ground_truths, predictions, target_names=["Negative", "Positive"], labels=[0, 1], digits=4)
    else:
        report = "N/A - No valid predictions parsed."

    print("\n" + "="*30)
    print(f"EVAL RESULTS: {SUBSET_TYPE}")
    print(f"Mode: {'Base Pre-trained' if USE_BASE_MODEL else 'Fine-tuned'}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("Classification Report:")
    print(report)
    print("="*30)

    # Save results
    final_output = {
        "config": {
            "is_base_model": USE_BASE_MODEL,
            "adapter_path": ADAPTER_PATH if not USE_BASE_MODEL else "None",
            "subset": SUBSET_TYPE,
            "test_index_path": TEST_INDEX_PATH
        },
        "metrics": {
            "accuracy": acc,
            "classification_report": report
        },
        "samples": results_data[:50] 
    }
    
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()