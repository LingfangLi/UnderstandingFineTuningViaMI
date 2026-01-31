import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# 1. Configuration
MODEL_TYPE = "llama"  # Options: "gpt2" or "llama"

if MODEL_TYPE == "gpt2":
    CONFIG = {
        "model_name": "gpt2",
        "checkpoint_path": "<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/gpt2-sst2.pt",
        "dtype": torch.float32,
        "padding_side": "right"
    }
else:
    CONFIG = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "checkpoint_path": "<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama3.2-sst2.pt",
        "dtype": torch.float16, # Llama recommends half-precision
        "padding_side": "left"   # Llama generation must use left padding
    }

CONFIG.update({
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "eval_base_model": False,
    "max_samples": 100 
})

# 2. Model Loading
def load_model_and_tokenizer():
    print(f"Loading {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    tokenizer.padding_side = CONFIG["padding_side"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained(
        CONFIG['model_name'],
        device=CONFIG['device'],
        dtype=CONFIG['dtype']
    )

    if not CONFIG["eval_base_model"]:
        print(f"Loading weights from {CONFIG['checkpoint_path']}...")
        state_dict = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, tokenizer

# 3. Data and Inference
model, tokenizer = load_model_and_tokenizer()
raw_data = load_dataset('yelp_polarity', split='test').select(range(CONFIG["max_samples"]))

# Label mapping (matches fine-tuning format)
label_map = {0: "Negative", 1: "Positive"}

count = 0
valid_count = 0

print(f"\nStarting Eval on {CONFIG['model_name']}...")
print(f"Max Context Length: {model.cfg.n_ctx}")

for i in range(len(raw_data)):
    try:
        sample = raw_data[i]
        prompt = f"Review: {sample['text']}\nSentiment:"
        true_label = label_map[sample["label"]]

        # Reserve 10 tokens for generation
        max_input_len = model.cfg.n_ctx - 10
        input_ids = tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_input_len
        ).to(CONFIG['device'])

        # Clamp out-of-vocabulary token IDs
        if torch.any(input_ids >= model.cfg.d_vocab):
            input_ids[input_ids >= model.cfg.d_vocab] = tokenizer.eos_token_id

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=5, 
                verbose=False,
                stop_at_eos=True
            )
        
        generated_part = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        if true_label.lower() in generated_part.lower():
            count += 1
        
        valid_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(raw_data)}] Accuracy: {count/valid_count:.4f}")

    except Exception as e:
        if "device-side assert" in str(e):
            print(f"FATAL ERROR: CUDA Assert at index {i}. Needs restart.")
            break
        print(f"Error at index {i}: {e}")

print("-" * 30)
print(f"Final Results for {CONFIG['model_name']}:")
print(f"Accuracy: {count/valid_count if valid_count > 0 else 0:.4f}")