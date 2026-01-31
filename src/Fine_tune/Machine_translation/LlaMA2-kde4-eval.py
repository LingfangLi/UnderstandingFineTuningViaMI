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
adapter_path = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/llama2-kde4-tech-trans-20251120-205734/checkpoint-1688/"

START_IDX = 30000
NUM_SAMPLES = 1000
BATCH_SIZE = 8

# Model Loading
print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # bfloat16 avoids inf/nan issues and is faster
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.padding_side = "left"  # Left padding required for generation
tokenizer.pad_token = tokenizer.eos_token

# Data Preparation
print("Loading Dataset...")
dataset = load_dataset('kde4', lang1="en", lang2="fr", trust_remote_code=True)['train'].select(
    range(START_IDX, START_IDX + NUM_SAMPLES))


def generate_prompt(english_text):
    return (f"Translate Technical English to French.\n\n"
            f"### Technical English:\n{english_text}\n\n"
            f"### Technical French:\n")


# Batch Evaluation Loop
bleu_scores = []
smoothing = SmoothingFunction().method1

print(f"Starting BATCH evaluation on {len(dataset)} samples...")

# Collect all source and reference texts
all_src = [sample['translation']['en'] for sample in dataset]
all_ref = [sample['translation']['fr'] for sample in dataset]

# Iterate in batches
for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    # Prepare batch data
    batch_src = all_src[i: i + BATCH_SIZE]
    batch_ref = all_ref[i: i + BATCH_SIZE]

    # Construct prompts
    batch_prompts = [generate_prompt(txt) for txt in batch_src]

    # Batch tokenization with automatic padding
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

    # Batch generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=False  # Greedy decoding
        )

    # Decode generated tokens (skip prompt portion)
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Compute metrics
    for pred_text, ref_text in zip(decoded_preds, batch_ref):
        # Post-processing
        clean_pred = pred_text.strip().split('\n')[0]

        # Calculate BLEU
        ref_tokens = [ref_text.split()]
        pred_tokens = clean_pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)

# Results
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
