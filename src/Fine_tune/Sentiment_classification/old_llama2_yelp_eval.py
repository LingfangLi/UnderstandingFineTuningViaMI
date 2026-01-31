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
ADAPTER_PATH = "<MODEL_STORAGE>/fine-tuning-project/old_fine_tuned_model/llama2-7b-sst2-qlora/"

BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "yelp_polarity"

TEST_SAMPLE_LIMIT = 1000

# 2. Load Model
print("Loading Base Model...")
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

print(f"Loading Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Prepare Data
print("Loading Yelp Test Dataset...")
test_dataset = load_dataset(DATASET_NAME, split="test")

if TEST_SAMPLE_LIMIT and len(test_dataset) > TEST_SAMPLE_LIMIT:
    print(f"Subsampling first {TEST_SAMPLE_LIMIT} examples for speed...")
    test_dataset = test_dataset.select(range(TEST_SAMPLE_LIMIT))

# 0 = Negative, 1 = Positive
id2label = {0: "negative", 1: "positive"}

# 4. Inference Loop
correct_count = 0
total_count = 0

print(f"Starting inference...")

for i, item in enumerate(tqdm(test_dataset)):
    text = item['text']
    
    # Truncate long reviews
    if len(text) > 1500:
        text = text[:1500]
        
    true_label_id = item['label']
    true_label_str = id2label[true_label_id]
    
    # Construct prompt (matches training format, without the label)
    prompt = f"Review: {text}\nSentiment:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
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
    except:
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
        print(f"Prompt (truncated): {prompt[:100]}...")
        print(f"Generated: {answer}")
        print(f"Pred: {pred_label} | True: {true_label_str}")

# 5. Results
accuracy = correct_count / total_count
print(f"\n{'='*30}")
print(f"Yelp Final Accuracy: {accuracy:.2%}")
print(f"Correct: {correct_count} / {total_count}")
print(f"{'='*30}")