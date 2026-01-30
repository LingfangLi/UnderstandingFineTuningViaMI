import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# 1. Configuration
base_model_name = "meta-llama/Llama-2-7b-hf"

adapter_path = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-yelp-qlora-20251119-154850/llama2-yelp/"

NUM_SAMPLES = 1000
MAX_LENGTH = 512
FINETUNED = False

# Label mapping (must match training)
TARGET_POSITIVE = "positive"
TARGET_NEGATIVE = "negative"

# 2. Load Model (QLoRA)

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
if FINETUNED:
    print(f"Loading fintuned model")
    print(f"Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
else:
    print("Load pretrained model")
    model =  base_model
        
model.eval()

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# Left padding is required for generation tasks
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


# 4. Prompt Template
def generate_eval_prompt(text):
    """Constructs the evaluation prompt without the label for model completion."""
    return f"Review: {text}\nSentiment:"

# 5. Evaluation Loop
dataset = load_dataset('yelp_polarity', split='test').select(range(NUM_SAMPLES))

correct_count = 0
total_count = 0

print(f"Starting evaluation on {NUM_SAMPLES} samples...")

for sample in tqdm(dataset):
    text = sample['text']
    label_idx = sample['label']  # 1 or 0

    # 1 -> positive, 0 -> negative
    expected_label = TARGET_POSITIVE if label_idx == 1 else TARGET_NEGATIVE

    prompt = generate_eval_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # Decode only newly generated tokens
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    pred_text = generated_text.strip().lower()
    print("text:::",text)
    print("Prediction:::",pred_text)
    print("label:::", expected_label)
    if expected_label in pred_text:
        correct_count += 1
    else:
        if total_count % 100 == 0 and total_count > 0:
            pass
            # print(f"\n[Fail] Expect: {expected_label} | Got: '{pred_text}'")

    total_count += 1

# 6. Output Results
accuracy = correct_count / total_count
print("-" * 30)
print(f"Total Samples: {total_count}")
print(f"Correct: {correct_count}")
print(f"Accuracy: {accuracy:.2%}")
print("-" * 30)