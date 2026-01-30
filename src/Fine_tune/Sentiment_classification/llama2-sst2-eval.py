# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# 1. Configuration
ADAPTER_PATH = ""

BASE_MODEL_NAME = "fadliaulawi/Llama-2-7b-finetuned"

TOKENIZER_NAME = "NousResearch/Llama-2-7b-hf"

DATASET_NAME = "stanfordnlp/sst2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FINETUNED = False

# SST-2 mapping: label 0 = negative, label 1 = positive
ID2LABEL = {
    0: "negative",
    1: "positive"
}

# 2. Load Model & Tokenizer
print(f"Loading Base Model: {BASE_MODEL_NAME} ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

#tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
print(f"Loading Tokenizer from: {TOKENIZER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.pad_token_id

if FINETUNED:
    print(f"Loading LoRA Adapter from: {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
else:
    print("Using Base Model only (Expect poor performance without few-shot)...")
    model = base_model
model.eval()

# 3. Load Test Dataset
dataset = load_dataset(DATASET_NAME, split="validation")
print(f"Validation set size: {len(dataset)}")

# 4. Prediction Function
def predict_sentiment(texts):
    prompts = [f"Review: {text}\nSentiment:" for text in texts]
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    full_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = []
    for prompt, response in zip(prompts, full_responses):
        generated_text = response[len(prompt):].strip().lower()

        if "positive" in generated_text:
            predictions.append(1)
        elif "negative" in generated_text:
            predictions.append(0)
        else:
            predictions.append(-1)
            
    return predictions

# 5. Inference Loop
batch_size = 16
all_predictions = []
all_labels = dataset['label']
all_texts = dataset['sentence']

print(f"Running inference (Batch Size: {batch_size})...")
for i in tqdm(range(0, len(all_texts), batch_size)):
    batch_texts = all_texts[i : i + batch_size]
    batch_preds = predict_sentiment(batch_texts)
    all_predictions.extend(batch_preds)

# 6. Evaluation
clean_preds = []
clean_labels = []

parse_error_count = 0
for p, l in zip(all_predictions, all_labels):
    if p != -1:
        clean_preds.append(p)
        clean_labels.append(l)
    else:
        parse_error_count += 1

print("\n" + "="*30)
print("Evaluation Results")
print("="*30)
print(f"Unparsable outputs: {parse_error_count} / {len(all_labels)}")

if len(clean_preds) > 0:
    acc = accuracy_score(clean_labels, clean_preds)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    target_names = ["Negative", "Positive"]
    print(classification_report(clean_labels, clean_preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(clean_labels, clean_preds))
else:
    print("No valid predictions. Check adapter path or training convergence.")

# 7. Demo
print("\n=== Demo Examples ===")
demo_indices = [0, 10, 20]
for idx in demo_indices:
    text = dataset[idx]['sentence']
    true_label = dataset[idx]['label']
    
    pred_label_id = predict_sentiment([text])[0]
    pred_str = ID2LABEL.get(pred_label_id, "Unknown")
    true_str = ID2LABEL[true_label]
    
    print(f"Review: {text}")
    print(f"True: {true_str} | Pred: {pred_str}")
    print("-" * 20)