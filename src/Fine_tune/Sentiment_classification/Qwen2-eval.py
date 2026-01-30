import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# 1. Configuration
model_path = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/qwen2-0.5b-full-ft-20251124-204027/checkpoint-1857/"

NUM_SAMPLES = 1000 
MAX_LENGTH = 512

TARGET_POSITIVE = "positive"
TARGET_NEGATIVE = "negative"

FINETUNED = False

# 2. Load Model
if FINETUNED:
    print(f"Loading Fine-Tuned Qwen2 model from {model_path}...")
else:
    print(f"Loading pretrained Qwen2 model from {model_path}...")
    model_path="Qwen/Qwen2-0.5B"

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
            
model.eval()

# 3. Load Tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Use left padding for generation tasks
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Prompt Template
def generate_eval_prompt(text):
    # Match training format: "Review: {text}\nSentiment: {label}"
    return f"Review: {text}\nSentiment:"

# 5. Evaluation Loop
dataset = load_dataset('yelp_polarity', split='test').select(range(NUM_SAMPLES))

correct_count = 0
total_count = 0

print(f"Starting evaluation on {NUM_SAMPLES} samples...")

for sample in tqdm(dataset):
    text = sample['text']
    label_idx = sample['label']  # 1 (pos) or 0 (neg)

    # Get expected answer
    expected_label = TARGET_POSITIVE if label_idx == 1 else TARGET_NEGATIVE

    # Construct input
    prompt = generate_eval_prompt(text)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,    # Use greedy search for classification
            temperature=1.0,
            use_cache=True
        )

    # Decode only the newly generated part
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Clean and compare
    pred_text = generated_text.strip().lower()
    print(f"Promt: ... | Pred: {pred_text} | Expect: {expected_label}")
    # Simple check: if expected label appears in prediction
    if expected_label in pred_text:
        correct_count += 1
    
    total_count += 1

# 6. Output Results
accuracy = correct_count / total_count
print("=" * 30)
print(f"Model: {model_path}")
print(f"Total Samples: {total_count}")
print(f"Correct: {correct_count}")
print(f"Accuracy: {accuracy:.2%}")
print("=" * 30)