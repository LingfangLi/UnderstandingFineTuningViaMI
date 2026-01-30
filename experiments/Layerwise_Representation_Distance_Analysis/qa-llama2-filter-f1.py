import json
import os
import string
import re
import collections
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Configuration
RESULTS_DIR = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/Results/QA/Llama2/20260118_224706/"
JSON_FILE = "batch_1_summary.json"

def normalize_text(s):
    """Remove articles, punctuation, convert to lowercase for lenient matching."""
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

def main():
    json_path = os.path.join(RESULTS_DIR, JSON_FILE)
    print(f"Loading results from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples. Reloading SQuAD dataset to get ALL gold answers...")

    dataset = load_dataset('squad')['validation']

    improved_count = 0

    print("Recalculating F1 scores using official evaluation logic...")

    for item in tqdm(data):
        idx = item['sample_idx']

        # Use all gold answers for max-F1 evaluation
        gold_answers = dataset[idx]['answers']['text']

        pred_before = item['prediction_before']
        pred_after = item['prediction_after']

        # Official logic: take highest score matching any gold answer
        f1_before = max(compute_f1(pred_before, gold) for gold in gold_answers)
        f1_after = max(compute_f1(pred_after, gold) for gold in gold_answers)

        item['f1_before'] = f1_before
        item['f1_after'] = f1_after

        if f1_after > f1_before:
            improved_count += 1

    new_json_path = os.path.join(RESULTS_DIR, "batch_1_summary_corrected.json")
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("-" * 40)
    print(f"Done! Updated results saved to: {new_json_path}")
    print(f"Samples with F1 improvement (Strict Eval): {improved_count}/{len(data)}")
    print("-" * 40)
    print("Now run your filter script pointing to this NEW json file.")
    print("(You may need to rename it to batch_1_summary.json or modify the filter script to read _corrected.json)")

if __name__ == "__main__":
    main()
