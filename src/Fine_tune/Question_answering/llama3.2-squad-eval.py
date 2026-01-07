import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import collections
import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama3.2-1b-SQUAD-full-ft-20260106-222423/checkpoint-5202/"
NUM_SAMPLES = 1000
MAX_LENGTH = 1024

# ==========================================
# 2. Helper Functions
# ==========================================
def normalize_text(s):
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
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

# ==========================================
# 3. Model Loading
# ==========================================
print(f"Loading Llama3.2 SQuAD model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 4. Data Preparation
# ==========================================
print("Preparing SQuAD validation set...")
raw_dataset = load_dataset('squad', split='validation').select(range(NUM_SAMPLES))

def generate_prompt(context, question):
    return (f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n")

# ==========================================
# 5. Evaluation Loop
# ==========================================
em_scores, f1_scores, bleu_scores = [], [], []
smoothing = SmoothingFunction().method1

print("Starting generation...")
for sample in tqdm(raw_dataset):
    context = sample['context']
    question = sample['question']
    gold_answers = sample['answers']['text']

    prompt = generate_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    # Decode
    input_len = inputs['input_ids'].shape[1]
    pred_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    pred_text = pred_text.strip().split('\n')[0]

    # Metrics
    sample_em = max(compute_exact_match(pred_text, gold) for gold in gold_answers)
    sample_f1 = max(compute_f1(pred_text, gold) for gold in gold_answers)
    ref_tokens_list = [ans.split() for ans in gold_answers]
    sample_bleu = sentence_bleu(ref_tokens_list, pred_text.split(), smoothing_function=smoothing)

    em_scores.append(sample_em)
    f1_scores.append(sample_f1)
    bleu_scores.append(sample_bleu)

print("\n" + "=" * 30)
print(f"GPT-2 SQuAD RESULTS (N={len(raw_dataset)})")
print(f"EM:   {np.mean(em_scores)*100:.2f}%")
print(f"F1:   {np.mean(f1_scores)*100:.2f}%")
print(f"BLEU: {np.mean(bleu_scores):.4f}")
print("=" * 30)