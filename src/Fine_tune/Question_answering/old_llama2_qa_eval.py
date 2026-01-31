import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
import string
import collections
import numpy as np

# 1. Configuration
CONFIG = {
    "task_name": "coqa",
    "adapter_path": "<MODEL_STORAGE>/fine-tuning-project/old_fine_tuned_model/llama2-7b-yelp-qlora/", #"<MODEL_STORAGE>/fine-tuning-project/old_fine_tuned_model/llama2-7b-coqa-manual-3000-qlora-20260122-193545", #
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eval_sample_limit": 1000, # None = full dataset
    "max_new_tokens": 50,
}

# 2. Metric Calculation (Standard QA Metrics)
def normalize_answer(s):
    """Remove punctuation, articles, and lowercase for normalized matching."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
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

# 3. Load Model
print(f"Loading Base Model: {CONFIG['base_model']}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG['base_model'],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading Adapter from: {CONFIG['adapter_path']}...")
model = PeftModel.from_pretrained(model, CONFIG['adapter_path'])
model.eval()

tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. Prepare Data
print(f"Loading {CONFIG['task_name']} Validation Dataset...")

eval_data = []

if CONFIG['task_name'] == 'squad':
    raw_data = load_dataset("squad", split="validation")
    if CONFIG['eval_sample_limit']:
        raw_data = raw_data.select(range(min(CONFIG['eval_sample_limit'], len(raw_data))))
        
    for item in raw_data:
        # Must match training prompt format
        prompt = f"Context: {item['context']} Question: {item['question']} Answer:"
        references = item['answers']['text']
        eval_data.append((prompt, references))

elif CONFIG['task_name'] == 'coqa':
    raw_data = load_dataset("stanfordnlp/coqa", split="validation")
    if CONFIG['eval_sample_limit']:
        raw_data = raw_data.select(range(min(CONFIG['eval_sample_limit'], len(raw_data))))
        
    for item in raw_data:
        story = item["story"]
        questions = item["questions"]
        answers = item["answers"]["input_text"]
        
        # CoQA flatten logic
        for q, a in zip(questions, answers):
            prompt = f"Context: {story} Question: {q} Answer:"
            references = [a]
            eval_data.append((prompt, references))

# Trim CoQA list if it exploded due to flattening
if CONFIG['eval_sample_limit'] and len(eval_data) > CONFIG['eval_sample_limit']:
    eval_data = eval_data[:CONFIG['eval_sample_limit']]
    
print(f"Prepared {len(eval_data)} samples for evaluation.")

# 5. Inference & Evaluation Loop (Batched)
def smart_clean(prediction, context):
    """Clean model predictions by truncating hallucinated content and aligning to context."""
    prediction = prediction.strip()

    # Truncate at common hallucination markers
    stop_phrases = ["Question:", "Context:", "Explanation:", "Note:", "3.", "4.", "\n"]
    for phrase in stop_phrases:
        if phrase in prediction:
            prediction = prediction.split(phrase)[0]

    prediction = prediction.strip()

    # Context alignment: find longest valid substring present in context
    if not prediction:
        return ""

    if prediction in context:
        return prediction

    # Find longest prefix that appears in context
    for i in range(len(prediction), 2, -1):
        sub = prediction[:i].strip()
        if sub in context:
            return sub

    return prediction

import torch

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left" 

def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
BATCH_SIZE = 16 
total_f1 = 0
total_em = 0
count = 0
# Evaluation Loop
print(f"Starting Inference with Smart Cleaning...")

for batch in tqdm(list(batch_list(eval_data, BATCH_SIZE))):

    batch_prompts = [item[0] for item in batch]
    batch_references = [item[1] for item in batch]

    # Extract context from prompt for alignment checking
    batch_contexts = []
    for p in batch_prompts:
        try:
            c_txt = p.split("Context:")[1].split("Question:")[0].strip()
            batch_contexts.append(c_txt)
        except:
            batch_contexts.append("")

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG['max_new_tokens'],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_length:]
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for prediction, references, context_text in zip(decoded_preds, batch_references, batch_contexts):
        final_pred = smart_clean(prediction, context_text)

        cur_f1 = max(compute_f1(final_pred, ref) for ref in references)
        cur_em = max(compute_exact(final_pred, ref) for ref in references)

        total_f1 += cur_f1
        total_em += cur_em
        count += 1

    #if count <= 5:
        print(f"\n[Fixed Debug]")
            
        current_raw = prediction.split('\n')[0] 
        print(f"Raw:     {current_raw}")
        print(f"Cleaned: {final_pred}")
        print(f"Ref:     {references}")
# 6. Final Results
avg_f1 = total_f1 / count if count > 0 else 0
avg_em = total_em / count if count > 0 else 0

print(f"\n{'='*30}")
print(f"Task: {CONFIG['task_name']}")
print(f"Samples: {count}")
print(f"Avg F1: {avg_f1:.2%}")
print(f"Avg EM: {avg_em:.2%}")
print(f"{'='*30}")