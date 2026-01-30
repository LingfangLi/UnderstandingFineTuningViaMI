import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# Config
MODEL_CONFIG = {
    "type": "llama",  # "gpt2" or "llama"
    "name": "meta-llama/Llama-3.2-1B",  # or "gpt2" 
    "checkpoint": "/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/llama3.2-sst2.pt",
    "batch_size": 16,
    "num_samples": 1000,
    "max_length": 256 
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class MTDataset(Dataset):
    def __init__(self, raw_data):
        self.samples = []
        for x in raw_data:
            en_text = x["translation"]["en"]
            fr_text = x["translation"]["fr"]
            prompt = f"Translate English to French. English: {en_text}\nFrench:"
            self.samples.append({"prompt": prompt, "target": fr_text})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


# Load Model
def load_mt_model():
    print(f"Loading tokenizer: {MODEL_CONFIG['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['name'])

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load HookedTransformer
    model = HookedTransformer.from_pretrained(
        MODEL_CONFIG['name'],
        device=DEVICE,
        dtype=torch.float16 if MODEL_CONFIG["type"] == "llama" else torch.float32
    )

    if MODEL_CONFIG["checkpoint"]:
        print(f"Loading model weight: {MODEL_CONFIG['checkpoint']}")
        state_dict = torch.load(MODEL_CONFIG["checkpoint"], map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, tokenizer


# Inference
def run_mt_eval():
    model, tokenizer = load_mt_model()

    print("Loading dataset...")
    raw_data = load_dataset('tatoeba', lang1="en", lang2="fr", split='train').select(
        range(40000, 40000 + MODEL_CONFIG["num_samples"]))
    dataset = MTDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=False)

    smoothing = SmoothingFunction().method1
    sum_bleu = 0
    processed_count = 0

    print(f"Start inference in | Batch Size: {MODEL_CONFIG['batch_size']}")

    for batch in tqdm(dataloader):
        prompts = batch["prompt"]
        targets = batch["target"]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODEL_CONFIG["max_length"]
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                inputs['input_ids'],
                max_new_tokens=50, 
                verbose=False,
                stop_at_eos=True,
                top_k=50,
                temperature=1.0  
            )

        input_len = inputs['input_ids'].shape[1]
        for i in range(len(prompts)):

            gen_tokens = output_ids[i][input_len:]
            prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            prediction = prediction.split("French:")[-1].strip()
            target = targets[i].strip()

            # Calculate BLEU
            score = sentence_bleu(
                [target.lower().split()],
                prediction.lower().split(),
                weights=(0.25, 0.25, 0, 0),  
                smoothing_function=smoothing
            )
            sum_bleu += score
            processed_count += 1

    avg_bleu = sum_bleu / processed_count
    print("\n" + "=" * 30)
    print(f"Model: {MODEL_CONFIG['name']}")
    print(f"Yask: English to French Translation")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    run_mt_eval()