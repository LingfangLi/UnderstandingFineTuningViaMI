"""Full-FT Llama-2-7B SQuAD evaluator on a lexical-complexity-filtered test
subset.

Counterpart of: llama2-qa-eval.py (LoRA version).

Required env vars:
  MODEL_PATH    path to a full FT model directory
  TRAIN_SUBSET  simple | complex
  TEST_SUBSET   simple | complex
Optional:
  OUTPUT_DIR    where to write the JSON
  IS_BASE_MODEL "1" to evaluate base pre-trained model
"""
import os
import re
import json
import string
import collections

import torch
from datasets import load_dataset
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

TEST_INDEX_PATH = f"{PROJECT_ROOT}/experiments/text_complexity/matrix_analysis/squad_lexically_{TEST_SUBSET}_test_indices.txt"

if IS_BASE_MODEL:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"eval_results_squad_test_{TEST_SUBSET}_base_full.json")
else:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"eval_results_squad_train_{TRAIN_SUBSET}_test_{TEST_SUBSET}_full.json")


def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(re.compile(r'\b(a|an|the)\b', re.UNICODE), ' ', t)
    def white_space_fix(t):
        return ' '.join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(pred: str, truth: str) -> float:
    pt = normalize_answer(pred).split()
    tt = normalize_answer(truth).split()
    if not pt or not tt:
        return float(pt == tt)
    common = collections.Counter(pt) & collections.Counter(tt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(tt)
    return 2 * precision * recall / (precision + recall)


def compute_exact(pred: str, truth: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(truth))


def generate_prompt(context: str, question: str) -> str:
    return (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Answer:\n"
    )


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

    full_dataset = load_dataset("squad", split="train")
    eval_dataset = full_dataset.select(test_indices)
    print(f"Eval set size: {len(eval_dataset)}")

    total_f1, total_em = 0.0, 0
    samples = []

    for i, item in enumerate(tqdm(eval_dataset)):
        context = item['context']
        question = item['question']
        references = item['answers']['text']

        prompt = generate_prompt(context, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs.input_ids.shape[1]
        pred_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        pred_text = pred_text.strip().split('\n')[0]

        cur_f1 = max((compute_f1(pred_text, r) for r in references), default=0.0)
        cur_em = max((compute_exact(pred_text, r) for r in references), default=0)
        total_f1 += cur_f1
        total_em += cur_em

        samples.append({
            "question": question,
            "prediction": pred_text,
            "references": references,
            "f1": cur_f1,
            "em": cur_em,
        })

    avg_f1 = total_f1 / len(eval_dataset)
    avg_em = total_em / len(eval_dataset)

    print("\n" + "=" * 32)
    print(f"SQuAD eval | train={TRAIN_SUBSET if not IS_BASE_MODEL else 'BASE'} | test={TEST_SUBSET}")
    print(f"avg_f1={avg_f1:.4f} | avg_em={avg_em:.4f}")
    print("=" * 32)

    final = {
        "config": {
            "is_base_model": IS_BASE_MODEL,
            "model_path": load_path,
            "train_subset": None if IS_BASE_MODEL else TRAIN_SUBSET,
            "test_subset": TEST_SUBSET,
            "test_index_path": TEST_INDEX_PATH,
        },
        "metrics": {"avg_f1": avg_f1, "avg_em": avg_em},
        "samples": samples,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
