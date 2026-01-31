import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configuration
base_model_name = "meta-llama/Llama-2-7b-hf"
adapter_path = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/llama2-tatoeba-en-fr-20251120-101824/checkpoint-4500/"

# Test samples (held-out data beyond training set)
START_IDX = 40000
NUM_SAMPLES = 1000

# Model Loading (QLoRA)
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading Adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# Left padding required for generation
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Data Preparation
print("Loading Tatoeba dataset (held-out test set)...")
# Load 1000 samples starting at index 40000 (unseen by the model)
dataset = load_dataset('tatoeba', lang1="en", lang2="fr", trust_remote_code=True)['train'].select(
    range(START_IDX, START_IDX + NUM_SAMPLES))


def generate_prompt(english_text):
    # Must match training format exactly, with blank French section
    return (f"Translate English to French.\n\n"
            f"### English:\n{english_text}\n\n"
            f"### French:\n")


# Evaluation Loop
bleu_scores = []
smoothing = SmoothingFunction().method1  # Prevents zero score on short sentences

print(f"Starting evaluation on {len(dataset)} samples...")

for i, sample in enumerate(tqdm(dataset)):
    # Extract source text and reference translation
    src_text = sample['translation']['en']
    ref_text = sample['translation']['fr']

    # Construct input
    prompt = generate_prompt(src_text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # Decode (extract only generated tokens)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Post-processing: strip whitespace and take only first line
    pred_text = pred_text.strip().split('\n')[0]

    # Calculate BLEU (NLTK expects references as list of lists)
    ref_tokens = [ref_text.split()]
    pred_tokens = pred_text.split()

    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
    bleu_scores.append(score)

    # Debug print (first 5 samples)
    if i < 5:
        print(f"\nSrc:  {src_text}")
        print(f"Ref:  {ref_text}")
        print(f"Pred: {pred_text}")
        print(f"BLEU: {score:.4f}")

# Results
avg_bleu = sum(bleu_scores) / len(bleu_scores)

print("\n" + "=" * 30)
print(f"EVALUATION RESULTS (N={NUM_SAMPLES})")
print("=" * 30)
print(f"Average BLEU Score: {avg_bleu:.4f}")
print("=" * 30)