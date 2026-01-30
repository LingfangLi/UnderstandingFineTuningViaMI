import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import collections
import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 1. Configuration & Paths
base_model_name = "meta-llama/Llama-2-7b-hf"

adapter_path = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-squad-qlora-20251119-214256/checkpoint-2476/"

NUM_SAMPLES = 1000
MAX_LENGTH = 1024


# 2. Helper Functions (SQuAD Standard Metrics)
def normalize_text(s):
    """Remove articles, punctuation, and lowercase for lenient matching."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# 3. Model Loading
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
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# 4. Data Preparation
print("Preparing dataset...")
raw_dataset = load_dataset('squad', split='train').select(range(20000,30000))

# Same seed as training ensures the test split contains unseen data
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
eval_dataset = dataset_dict['test'].select(range(NUM_SAMPLES))

print(f"Evaluating on {len(eval_dataset)} unseen samples.")


def generate_prompt(context, question):
    # Must match training prompt format (without answer)
    return (f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n")


# 5. Evaluation Loop
em_scores = []
f1_scores = []
bleu_scores = []
smoothing = SmoothingFunction().method1

print("Starting generation...")
for i, sample in enumerate(tqdm(eval_dataset)):
    context = sample['context']
    question = sample['question']
    gold_answers = sample['answers']['text']

    prompt = generate_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # Extract only the generated portion
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    pred_text = pred_text.strip()
    # Truncate at newline (model may generate continuation)
    pred_text = pred_text.split('\n')[0]

    # Best score across all reference answers
    sample_em = max(compute_exact_match(pred_text, gold) for gold in gold_answers)
    sample_f1 = max(compute_f1(pred_text, gold) for gold in gold_answers)

    ref_tokens_list = [ans.split() for ans in gold_answers]
    sample_bleu = sentence_bleu(ref_tokens_list, pred_text.split(), smoothing_function=smoothing)

    em_scores.append(sample_em)
    f1_scores.append(sample_f1)
    bleu_scores.append(sample_bleu)

    # Debug: print first 5 samples
    if i < 5:
        print(f"\nQ: {question}")
        print(f"Pred: {pred_text}")
        print(f"Gold: {gold_answers}")
        print(f"Scores -> EM: {sample_em}, F1: {sample_f1:.2f}, BLEU: {sample_bleu:.2f}")

# 6. Final Statistics
avg_em = np.mean(em_scores) * 100
avg_f1 = np.mean(f1_scores) * 100
avg_bleu = np.mean(bleu_scores)

print("\n" + "=" * 30)
print(f"EVALUATION RESULTS (N={len(eval_dataset)})")
print("=" * 30)
print(f"Exact Match (EM): {avg_em:.2f}%")
print(f"F1 Score:         {avg_f1:.2f}%")
print(f"BLEU Score:       {avg_bleu:.4f}")
print("=" * 30)