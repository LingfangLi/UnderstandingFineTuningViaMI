import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = "/users/sglli24/UnderstandingFineTuningViaMI/src/Fine_tune/Sentiment_classification/fine_tuned_model/gpt2-small-yelp-full-ft-20260105-175008/checkpoint-930/" 
DATASET_NAME = "yelp_polarity"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED = False

def generate_prompt(text):
    return f"Review: {text}\nSentiment:"

def main():
    if FINETUNED:
        print(f"Loading model from: {MODEL_PATH} ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,torch_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    dataset = load_dataset(DATASET_NAME, split="test").select(range(1000))
    
    correct = 0
    total = 0
    
    batch_size = 8
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        prompts = [generate_prompt(t) for t in batch['text']]
        # 0->negative, 1->positive
        labels = ["positive" if l == 1 else "negative" for l in batch['label']]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():

            outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
            
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for pred, label in zip(generated_texts, labels):
            pred_clean = pred.strip().lower()
            if label in pred_clean:
                correct += 1
            total += 1
            
    print(f"Accuracy: {correct/total:.2%}")

if __name__ == "__main__":
    main()