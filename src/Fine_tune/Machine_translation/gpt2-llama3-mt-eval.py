import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configuration
EVAL_BASE_MODEL = True

MODEL_TYPE = "llama3.2"  # "gpt2", "llama3.2"

if MODEL_TYPE == "gpt2":
    BASE_MODEL_ID = "gpt2"
    FT_MODEL_PATH = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/gpt2-small-kde4-full-ft-20260106-204426/gpt2-small-kde4-full-ft-20260106-204426/checkpoint-2532/"
elif MODEL_TYPE == "llama3.2":
    BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
    FT_MODEL_PATH = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/llama3.2-1b-kde4-full-checkpoint/"

MODEL_PATH = BASE_MODEL_ID if EVAL_BASE_MODEL else FT_MODEL_PATH

# Task settings
IS_KDE4 = True  # Set to False for Tatoeba, True for KDE4
DATASET_NAME = 'kde4' if IS_KDE4 else 'tatoeba'

START_IDX = 30000 if IS_KDE4 else 40000 
NUM_SAMPLES = 1000
BATCH_SIZE = 16 

print("="*40)
print(f"Testing Mode: {'[BASE MODEL]' if EVAL_BASE_MODEL else '[FINE-TUNED MODEL]'}")
print(f"Model Path: {MODEL_PATH}")
print(f"Task: {DATASET_NAME}")
print("="*40)

# Model Loading
print(f"Loading model...")
dtype = torch.bfloat16 if "llama" in MODEL_PATH.lower() and torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=dtype,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"

# Fix pad token if missing (common in GPT-2/Llama base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    if "llama" in MODEL_PATH.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id

# Data Preparation
print(f"Loading {DATASET_NAME} dataset...")
# Load dataset
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
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    # Batch generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False, # Greedy decoding for reproducibility
            temperature=1.0,
            use_cache=True
        )
    
    # Batch decoding
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Calculate metrics for each prediction
    for pred_text, ref_text in zip(decoded_preds, batch_ref):
        clean_pred = pred_text.strip().split('\n')[0]
        
        if i != 0: 
            print(f"\n[Debug Sample]\nRef: {ref_text}\nPred: {clean_pred}\n")
            i = -1 # prevent spamming

        ref_tokens = [ref_text.split()]
        pred_tokens = clean_pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)

# Results
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print("\n" + "=" * 40)
print(f"Model: {MODEL_PATH}")
print(f"Type: {'BASE' if EVAL_BASE_MODEL else 'FINE-TUNED'}")
print(f"Task: {DATASET_NAME}")
print(f"Samples: {NUM_SAMPLES}")
print(f"Average BLEU Score: {avg_bleu:.4f}")
print("=" * 40)