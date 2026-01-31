import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# Configuration
CONFIG = {
    "task_name": "kde4",

    "adapter_path": "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/llama2-7b-yelp-qlora/",

    "base_model": "meta-llama/Llama-2-7b-hf",

    "batch_size": 32,
    
    # Data limits consistent with training code
    "train_limit_kde4": 30000,
    "train_limit_tatoeba": 40000,
    
    "max_new_tokens": 128,
    "seed": 42
}

# Dataset Class
class MTEvalDataset(Dataset):
    def __init__(self, data_source):
        self.samples = []
        
        # Training prompt: "Translate English to French. English: {en}\nFrench: {fr}"
        # Testing prompt:  "Translate English to French. English: {en}\nFrench:"
        print(f"Formatting {len(data_source)} evaluation samples...")
        
        for item in data_source:
            en_text = item['translation']['en']
            fr_text = item['translation']['fr']
            
            prompt = f"Translate English to French. English: {en_text}\nFrench:"
            self.samples.append((prompt, fr_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Model and Tokenizer Loader
def load_model():
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

    print(f"Loading Adapter: {CONFIG['adapter_path']}...")
    #model = PeftModel.from_pretrained(model, CONFIG['adapter_path'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'], trust_remote_code=True)
    
    # Left padding is required for batch inference
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    return model, tokenizer

# Data Preparation (Strict Alignment)
def get_eval_dataloader():
    if CONFIG['task_name'] == 'kde4':
        print("Loading KDE4 (Explicit Split)...")
        # Replicate training code: take segment from 30000 to 31000
        raw_data = load_dataset("kde4", lang1="en", lang2="fr", split="train")
        val_slice = raw_data.select(range(CONFIG['train_limit_kde4'], CONFIG['train_limit_kde4'] + 1000))
        dataset = MTEvalDataset(val_slice)
        
    elif CONFIG['task_name'] == 'tatoeba':
        print("Loading Tatoeba (Random Split Reconstruction)...")
        # Replicate training code: take first 40000, then apply random split
        raw_data = load_dataset("tatoeba", lang1="en", lang2="fr", split="train", trust_remote_code=True).select(range(CONFIG['train_limit_tatoeba']))
        
        train_size = int(0.9 * len(raw_data))
        val_size = len(raw_data) - train_size
        
        # Use the same seed to reproduce the split
        _, val_subset = random_split(raw_data, [train_size, val_size], generator=torch.Generator().manual_seed(CONFIG['seed']))
        dataset = MTEvalDataset(val_subset)
        
    return DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Evaluation Loop (Batched)
import argparse
def main():
    #parser = argparse.ArgumentParser(description="LLaMA MT Evaluation Launcher")
    #parser.add_argument("--task", type=str, required=True, help="Task name for log and config")
    #parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter")
    #args = parser.parse_args()

    #CONFIG['task_name'] = args.task  
    #CONFIG['adapter_path'] = args.adapter  

    print(f"Active Task: {CONFIG['task_name']}")
    print(f"Loading Adapter from: {CONFIG['adapter_path']}")
    model, tokenizer = load_model()
    dataloader = get_eval_dataloader()
    
    smoothing = SmoothingFunction().method1
    total_bleu = 0
    count = 0
    
    print(f"Starting Batched Evaluation (Batch Size: {CONFIG['batch_size']})...")
    
    for batch_prompts, batch_refs in tqdm(dataloader):
        # Tokenize batch (left padding)
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to("cuda")
        
      
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG['max_new_tokens'], 
                do_sample=False,        
                repetition_penalty=1.2, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 3. Decode & Calculate Metric
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for i, full_text in enumerate(decoded_outputs):
            prompt = batch_prompts[i]
            ref_text = batch_refs[i]
            
            
            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
            else:
                
                generated_text = full_text.split("French:")[-1].strip()
            
            generated_text = generated_text.split("\n")[0].strip()
            
            if "(" in generated_text:
                
                generated_text = generated_text.split("(")[0].strip()
            
            stop_words = ["Ref:", "English:", "Source:", "Attention"]
            for sw in stop_words:
                if sw in generated_text:
                    generated_text = generated_text.split(sw)[0].strip()


            ref_tokens = ref_text.lower().split()
            hyp_tokens = generated_text.lower().split()
            
            if len(hyp_tokens) > 0:
                score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            else:
                score = 0.0
                
            total_bleu += score
            count += 1
            
            
            if count <= 5: 
                print(f"\n[Sample {count}]")
                print(f"Ref: {ref_text}")
                print(f"Gen (Raw): {full_text[len(prompt):][:100]}...") 
                print(f"Gen (Cleaned): {generated_text}")
                print(f"BLEU: {score:.4f}")

    avg_bleu = total_bleu / count if count > 0 else 0
    
    print(f"\n{'='*30}")
    print(f"Task: {CONFIG['task_name']}")
    print(f"Model: Llama-2-7b + LoRA (Aggressive Cleaning)")
    print(f"Samples Evaluated: {count}")
    print(f"Avg BLEU: {avg_bleu:.4f}")
    print(f"{'='*30}")

if __name__ == "__main__":
    main()