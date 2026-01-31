# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# 1. Configuration
MODEL_PATH = "<PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/fine_tuned_model/gpt2-small-full-ft-20251205-172809/checkpoint-939/"
DATASET_NAME = "stanfordnlp/sst2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED = False

# SST-2 mapping: label 0 = negative, label 1 = positive
ID2LABEL = {
    0: "negative",
    1: "positive"
}

# 2. Load Model & Tokenizer
if FINETUNED:
    print(f"Loading model from: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
else:
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Left padding is required for generation
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

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
        max_length=512
    ).to(DEVICE)

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
batch_size = 32
all_predictions = []
all_labels = dataset['label']
all_texts = dataset['sentence']

print("Running inference...")
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
print(f"Unparsable outputs (model generated unexpected text): {parse_error_count} / {len(all_labels)}")

if len(clean_preds) > 0:
    acc = accuracy_score(clean_labels, clean_preds)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    target_names = ["Negative", "Positive"]
    print(classification_report(clean_labels, clean_preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(clean_labels, clean_preds))
else:
    print("No valid predictions. Check the prompt format or training quality.")

# 7. Demo
print("\n=== Demo Examples ===")
demo_indices = [0, 10, 20, 30, 40]
for idx in demo_indices:
    text = dataset[idx]['sentence']
    true_label = dataset[idx]['label']
    
    pred_label_id = predict_sentiment([text])[0]
    pred_str = ID2LABEL.get(pred_label_id, "Unknown")
    true_str = ID2LABEL[true_label]
    
    print(f"Review: {text}")
    print(f"True: {true_str} | Pred: {pred_str}")
    print("-" * 20)