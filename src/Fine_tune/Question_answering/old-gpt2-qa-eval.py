import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# Configuration
MODEL_CONFIG = {
    "type": "llama",  # "gpt2" or "llama"
    "name": "meta-llama/Llama-3.2-1B",
    "checkpoint": "<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama3.2-sst2.pt",
    "task": "coqa",  # "squad" oe "coqa" 
    "batch_size": 16,  
    "num_samples": 1000
}

MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Dataset
class QADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_qa_samples(task, num_samples=1000):
    if task == "squad":
        ds = load_dataset('squad', split='validation').select(range(num_samples))
        return [{"context": x["context"], "question": x["question"], "answer": x["answers"]["text"][0]} for x in ds]
    elif task == "coqa":
        ds = load_dataset('coqa', split='train')
        extracted = []
        count = 0
        start_pair = 36000
        for sample in ds:
            for q, a in zip(sample["questions"], sample["answers"]["input_text"]):
                count += 1
                if count > start_pair:
                    extracted.append({"context": sample["story"], "question": q, "answer": a})
                if len(extracted) >= num_samples: return extracted
        return extracted


# 2. Main Inference
def run_eval():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['name'])
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained(
        MODEL_CONFIG['name'], device=DEVICE,
        dtype=torch.float16 if "llama" in MODEL_CONFIG["type"] else torch.float32
    )
    if MODEL_CONFIG["checkpoint"]:
        state_dict = torch.load(MODEL_CONFIG["checkpoint"], map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    samples = get_qa_samples(MODEL_CONFIG["task"], MODEL_CONFIG["num_samples"])
    dataset = QADataset(samples)
    dataloader = DataLoader(dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=False)

    smoothing = SmoothingFunction().method1
    total_bleu = 0
    correct = 0

    print(f"Starting Batch Eval on {DEVICE}...")
    for batch in tqdm(dataloader):
        prompts = [f"Context: {c}\nQuestion: {q}\nAnswer:" for c, q in zip(batch["context"], batch["question"])]
        targets = batch["answer"]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH - 50).to(
            DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                inputs['input_ids'], max_new_tokens=30,
                verbose=False, stop_at_eos=True
            )

        input_len = inputs['input_ids'].shape[1]
        for i in range(len(prompts)):
            gen_tokens = output_ids[i][input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            target = targets[i].strip()

            bleu = sentence_bleu([target.lower().split()], gen_text.lower().split(),
                                 weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            total_bleu += bleu
            if gen_text.lower() == target.lower(): correct += 1

    print(f"\nFinal BLEU: {total_bleu / len(samples):.4f}")
    print(f"EM: {correct}/{len(samples)}")


if __name__ == "__main__":
    run_eval()