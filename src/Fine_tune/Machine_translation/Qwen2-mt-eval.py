import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configuration
model_path = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/qwen2-0.5b-tatoeba-en-fr-20251125-165129/"

# Task settings
IS_KDE4 = False  # Set to False for Tatoeba, True for KDE4
DATASET_NAME = 'kde4' if IS_KDE4 else 'tatoeba'

START_IDX = 30000 if IS_KDE4 else 40000
NUM_SAMPLES = 1000
BATCH_SIZE = 16  # Qwen 0.5B is fast, can use larger batches

# Model Loading
print(f"Loading Qwen2 model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = "left"  # Left padding required for generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Data Preparation
print(f"Loading {DATASET_NAME} dataset...")
dataset = load_dataset(DATASET_NAME, lang1="en", lang2="fr", trust_remote_code=True)['train'].select(
    range(START_IDX, START_IDX + NUM_SAMPLES))

def generate_prompt(english_text):
    if IS_KDE4:
        # KDE4 technical prompt
        return (f"Translate Technical English to French.\n\n"
                f"### Technical English:\n{english_text}\n\n"
                f"### Technical French:\n")
    else:
        # Tatoeba general prompt
        return (f"Translate English to French.\n\n"
                f"### English:\n{english_text}\n\n"
                f"### French:\n")

# Batch Evaluation
bleu_scores = []
smoothing = SmoothingFunction().method1

print(f"Starting BATCH evaluation on {len(dataset)} samples...")

all_src = [sample['translation']['en'] for sample in dataset]
all_ref = [sample['translation']['fr'] for sample in dataset]

# Process in batches
for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch_src = all_src[i: i + BATCH_SIZE]
    batch_ref = all_ref[i: i + BATCH_SIZE]

    batch_prompts = [generate_prompt(txt) for txt in batch_src]
    
    # Batch tokenization
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

    # Batch generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # Translations are typically short
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    # Batch decoding
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Calculate metrics for each prediction
    for pred_text, ref_text in zip(decoded_preds, batch_ref):
        clean_pred = pred_text.strip().split('\n')[0]
        
        ref_tokens = [ref_text.split()]
        pred_tokens = clean_pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)

# Results
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print("\n" + "=" * 30)
print(f"Task: {DATASET_NAME} | Model: Qwen2-0.5B")
print(f"Samples: {NUM_SAMPLES}")
print(f"Average BLEU Score: {avg_bleu:.4f}")
print("=" * 30)