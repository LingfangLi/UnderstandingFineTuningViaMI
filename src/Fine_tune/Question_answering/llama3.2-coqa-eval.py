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
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

# ==========================================
# 3. Model Loading
# ==========================================
print(f"Loading GPT-2 CoQA model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 4. Data Preparation (Flatten Logic)
# ==========================================
print("Loading CoQA validation set...")
raw_val_data = load_dataset('stanfordnlp/coqa', split='validation')

def flatten_coqa_val_with_history(dataset, limit=None):
    flattened_samples = []
    count = 0
    for sample in dataset:
        story = sample['story']
        questions = sample['questions']
        answers = sample['answers']['input_text']
        history_buffer = []

        for q, a in zip(questions, answers):
            # Construct history (consistent with training)
            if len(history_buffer) == 0:
                history_str = "None"
            else:
                recent_history = history_buffer[-5:]
                history_str = "\n".join(recent_history)
            
            flattened_samples.append({
                'context': story,
                'history': history_str,
                'question': q,
                'gold_answer': a
            })
            
            # Update history buffer
            history_buffer.append(f"User: {q}\nAssistant: {a}")
            
            count += 1
            if limit and count >= limit:
                return flattened_samples
    return flattened_samples

test_samples = flatten_coqa_val_with_history(raw_val_data, limit=NUM_SAMPLES)

def generate_prompt(context, history, question):
    return (f"### Context:\n{context}\n\n"
            f"### Chat History:\n{history}\n\n"
            f"### Current Question:\n{question}\n\n"
            f"### Current Answer:\n")

# ==========================================
# 5. Evaluation Loop
# ==========================================
em_scores, f1_scores, bleu_scores = [], [], []
smoothing = SmoothingFunction().method1

print("Starting generation...")
for sample in tqdm(test_samples):
    prompt = generate_prompt(sample['context'], sample['history'], sample['question'])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    input_len = inputs['input_ids'].shape[1]
    pred_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    pred_text = pred_text.strip().split('\n')[0]

    gold_answer = sample['gold_answer']
    em_scores.append(compute_exact_match(pred_text, gold_answer))
    f1_scores.append(compute_f1(pred_text, gold_answer))
    bleu_scores.append(sentence_bleu([gold_answer.split()], pred_text.split(), smoothing_function=smoothing))

print("\n" + "=" * 30)
print(f"GPT-2 CoQA RESULTS (N={len(test_samples)})")
print(f"EM:   {np.mean(em_scores)*100:.2f}%")
print(f"F1:   {np.mean(f1_scores)*100:.2f}%")
print(f"BLEU: {np.mean(bleu_scores):.4f}")
print("=" * 30)