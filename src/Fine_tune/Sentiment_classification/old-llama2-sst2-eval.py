import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# 1. Configuration
ADAPTER_PATH = "/mnt/scratch/users/sglli24/fine-tuning-project/old_fine_tuned_model/llama2-7b-tatoeba-qlora/"

BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "stanfordnlp/sst2"

# 2. Load Model & Tokenizer
print("Loading Base Model (Llama-2-7b) in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading LoRA Adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Load Validation Data
print("Loading SST-2 Validation Dataset...")
eval_dataset = load_dataset(DATASET_NAME, split="validation")
id2label = {0: "negative", 1: "positive"}

# 4. Inference Loop
correct_count = 0
total_count = 0

print(f"Starting evaluation on {len(eval_dataset)} samples...")

for i, item in enumerate(tqdm(eval_dataset)):
    text = item['sentence']
    true_label_id = item['label']
    true_label_str = id2label[true_label_id]
    
    prompt = f"Review: {text}\nSentiment:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        answer = generated_text.split("Sentiment:")[-1].strip().lower()
    except IndexError:
        answer = ""
    
    pred_label = "unknown"
    if "positive" in answer:
        pred_label = "positive"
    elif "negative" in answer:
        pred_label = "negative"
        
    if pred_label == true_label_str:
        correct_count += 1
    
    total_count += 1
    
    if i < 3:
        print(f"\n[Sample {i}]")
        print(f"Prompt:     {prompt.replace(chr(10), ' ')}")
        print(f"Generated:  {answer}")
        print(f"Prediction: {pred_label} | True: {true_label_str}")

# 5. Final Report
accuracy = correct_count / total_count
print(f"\n{'='*30}")
print(f"Final Accuracy: {accuracy:.2%}")
print(f"Correct: {correct_count} / {total_count}")
print(f"{'='*30}")

if accuracy > 0.85:
    print("Result: Excellent.")
elif accuracy > 0.50:
    print("Result: Moderate.")
else:
    print("Result: Poor.")