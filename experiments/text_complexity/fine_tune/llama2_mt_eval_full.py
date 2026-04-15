"""Full-FT Llama-2-7B Tatoeba (en->fr) evaluator on a lexical-complexity-
filtered test subset.

Counterpart of: llama2-mt-eval.py (LoRA version).

Required env vars:
  MODEL_PATH    path to a full FT model directory
  TRAIN_SUBSET  simple | complex
  TEST_SUBSET   simple | complex
Optional:
  OUTPUT_DIR    where to write the JSON
  IS_BASE_MODEL "1" to evaluate base pre-trained model
"""
import os
import json

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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

TEST_INDEX_PATH = f"{PROJECT_ROOT}/experiments/text_complexity/matrix_analysis/tatoeba_lexically_{TEST_SUBSET}_test_indices.txt"

if IS_BASE_MODEL:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"eval_results_tatoeba_test_{TEST_SUBSET}_base_full_nltk.json")
else:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"eval_results_tatoeba_train_{TRAIN_SUBSET}_test_{TEST_SUBSET}_full_nltk.json")


def generate_prompt(en_text: str) -> str:
    return f"Translate English to French. English: {en_text}\nFrench:"


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

    full_dataset = load_dataset("tatoeba", name="en-fr", lang1="en", lang2="fr", split="train", trust_remote_code=True)
    test_dataset = full_dataset.select(test_indices)
    print(f"Eval set size: {len(test_dataset)}")

    smoothing = SmoothingFunction().method1
    total_score = 0.0
    count = 0
    samples = []
    batch_size = 8

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i : i + batch_size]
        en_texts = [item['en'] for item in batch['translation']]
        fr_refs  = [item['fr'] for item in batch['translation']]

        prompts = [generate_prompt(t) for t in en_texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs.input_ids.shape[1]
        decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for en, ref_str, pred_str in zip(en_texts, fr_refs, decoded):
            pred_clean = pred_str.strip().split('\n')[0]
            ref_tokens = ref_str.lower().split()
            hyp_tokens = pred_clean.lower().split()
            score = (
                sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
                if hyp_tokens else 0.0
            )
            total_score += score
            count += 1
            samples.append({
                "english": en,
                "reference": ref_str,
                "prediction": pred_clean,
                "bleu": score,
            })

    avg_bleu = total_score / count if count > 0 else 0.0
    print("\n" + "=" * 32)
    print(f"Tatoeba eval | train={TRAIN_SUBSET if not IS_BASE_MODEL else 'BASE'} | test={TEST_SUBSET}")
    print(f"avg_bleu={avg_bleu:.4f}")
    print("=" * 32)

    final = {
        "config": {
            "is_base_model": IS_BASE_MODEL,
            "model_path": load_path,
            "train_subset": None if IS_BASE_MODEL else TRAIN_SUBSET,
            "test_subset": TEST_SUBSET,
            "test_index_path": TEST_INDEX_PATH,
        },
        "metrics": {"avg_bleu": avg_bleu},
        "samples": samples[:50],
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
