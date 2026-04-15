"""Full-FT Llama-2-7B Yelp evaluator on a lexical-complexity-filtered test
subset.

Counterpart of: llama-sentiment-eval.py (LoRA version).

Required env vars:
  MODEL_PATH    path to a full FT model directory (loaded with from_pretrained)
  TRAIN_SUBSET  simple | complex   (which subset the model was trained on)
  TEST_SUBSET   simple | complex   (which subset to evaluate on)
Optional:
  OUTPUT_DIR    where to write the JSON (default: cwd)
  IS_BASE_MODEL "1" to evaluate base pre-trained Llama-2-7B and ignore MODEL_PATH

Output filename:
  eval_results_yelp_train_<TRAIN>_test_<TEST>_full.json
  (or eval_results_yelp_test_<TEST>_base_full.json if IS_BASE_MODEL=1)
"""
import os
import json

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

_NEW_API = int(transformers.__version__.split('.')[0]) >= 5

PROJECT_ROOT = "/users/sglli24/UnderstandingFineTuningViaMI"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"

IS_BASE_MODEL = os.environ.get("IS_BASE_MODEL", "0") == "1"
MODEL_PATH = os.environ.get("MODEL_PATH", "")
TRAIN_SUBSET = os.environ.get("TRAIN_SUBSET", "")
TEST_SUBSET = os.environ.get("TEST_SUBSET", "simple")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.dirname(__file__))

assert TEST_SUBSET in ("simple", "complex"), f"TEST_SUBSET must be simple|complex, got {TEST_SUBSET}"
if not IS_BASE_MODEL:
    assert MODEL_PATH and os.path.isdir(MODEL_PATH), f"MODEL_PATH must be a valid dir; got {MODEL_PATH}"
    assert TRAIN_SUBSET in ("simple", "complex"), f"TRAIN_SUBSET must be simple|complex, got {TRAIN_SUBSET}"

TEST_INDEX_PATH = f"{PROJECT_ROOT}/experiments/text_complexity/matrix_analysis/yelp_lexically_{TEST_SUBSET}_test_indices.txt"

if IS_BASE_MODEL:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"eval_results_yelp_test_{TEST_SUBSET}_base_full.json")
else:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"eval_results_yelp_train_{TRAIN_SUBSET}_test_{TEST_SUBSET}_full.json")


def generate_prompt(text: str) -> str:
    clean = str(text).replace('\n', ' ').strip()
    return f"Review: {clean}\nSentiment:"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Mode: {'BASE pretrained' if IS_BASE_MODEL else f'FULL FT (train={TRAIN_SUBSET})'} | test={TEST_SUBSET}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_path = BASE_MODEL if IS_BASE_MODEL else MODEL_PATH
    print(f"Loading model from: {load_path}")
    _dtype_kwarg = {"dtype": torch.bfloat16} if _NEW_API else {"torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        **_dtype_kwarg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading test indices from {TEST_INDEX_PATH}")
    if not os.path.exists(TEST_INDEX_PATH):
        raise FileNotFoundError(TEST_INDEX_PATH)
    with open(TEST_INDEX_PATH) as f:
        test_indices = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(test_indices)} indices")

    raw_test = load_dataset("yelp_polarity", split="train").select(test_indices)
    print(f"Eval set size: {len(raw_test)}")

    predictions = []
    ground_truths = []
    samples_out = []

    batch_size = 8
    for i in tqdm(range(0, len(raw_test), batch_size)):
        batch = raw_test[i : i + batch_size]
        texts = batch['text']
        labels = batch['label']

        prompts = [generate_prompt(t) for t in texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        gen_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for text, pred_str, true_label in zip(texts, decoded, labels):
            pred_clean = pred_str.strip().lower()
            if "positive" in pred_clean:
                predicted_label = 1
            elif "negative" in pred_clean:
                predicted_label = 0
            else:
                predicted_label = -1
            predictions.append(predicted_label)
            ground_truths.append(true_label)
            if len(samples_out) < 5:
                samples_out.append({
                    "text_snippet": text[:100],
                    "prediction_raw": pred_str,
                    "prediction_mapped": predicted_label,
                    "ground_truth": true_label,
                    "correct": predicted_label == true_label,
                })

    acc = accuracy_score(ground_truths, predictions)
    if any(p != -1 for p in predictions):
        report = classification_report(
            ground_truths, predictions,
            target_names=["Negative", "Positive"], labels=[0, 1], digits=4,
        )
    else:
        report = "N/A - no valid predictions parsed."

    print("\n" + "=" * 32)
    print(f"Yelp eval | train={TRAIN_SUBSET if not IS_BASE_MODEL else 'BASE'} | test={TEST_SUBSET}")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("=" * 32)

    final = {
        "config": {
            "is_base_model": IS_BASE_MODEL,
            "model_path": load_path,
            "train_subset": None if IS_BASE_MODEL else TRAIN_SUBSET,
            "test_subset": TEST_SUBSET,
            "test_index_path": TEST_INDEX_PATH,
        },
        "metrics": {"accuracy": acc, "classification_report": report},
        "samples": samples_out,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
