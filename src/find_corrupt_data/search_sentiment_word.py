import re
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os

# ==========================================
# CONFIGURATION
# ==========================================
# [IMPORTANT] Update this path to the specific folder created by your training script
# Example: "/users/sglli24/fine-tuning-project/fine_tuned_model/gpt2-small-full-ft-20250101-120000"
TRAINED_MODEL_PATH = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/gpt2-sst2-full-ft-20251205-172809/"

# Check if path exists
if "YOUR_SPECIFIC_RUN_FOLDER_HERE" in TRAINED_MODEL_PATH or not os.path.exists(TRAINED_MODEL_PATH):
    print(f"Warning: Please set the correct TRAINED_MODEL_PATH. Current: {TRAINED_MODEL_PATH}")

# Step 1: Load the trained model (Switched from HookedTransformer to AutoModel to match training output)
print(f"Loading model from {TRAINED_MODEL_PATH}...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Model and Tokenizer from the training output directory
try:
    model = AutoModelForCausalLM.from_pretrained(
        TRAINED_MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Fallback: Loading base GPT2 (Logic will fail if model path is wrong)")
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

model.eval()

# Ensure padding token is set (consistent with training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load the test dataset
# Note: Training used SST2, here we test on Yelp (Cross-domain). 
test_data = load_dataset("stanfordnlp/sst2", split="validation")

# Step 3: Function to predict sentiment
# [UPDATED] Matches the training format: "Sentiment: positive" or "Sentiment: negative"
def predict_sentiment(model, tokenizer, text):
    """
    Predicts sentiment by checking the next token after "Sentiment: ".
    The training script formatted data as: "Review: {text}\nSentiment: {positive/negative}"
    """
    prompt = f"Review: {text}\nSentiment: " # Matches training format
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]  # Logits for the next token
    
    # Get Token IDs for " positive" and " negative" 
    # Note: Depending on tokenizer, space might be part of the token. 
    # Usually GPT2 adds a space before the word if it's not the start of a sentence.
    # Since prompt ends in space "Sentiment: ", the next token is likely just "positive".
    
    # We strip whitespace to be safe and encode simple words
    pos_id = tokenizer.encode("positive")[0]
    neg_id = tokenizer.encode("negative")[0]
    
    prob_pos = logits[0, pos_id].item()
    prob_neg = logits[0, neg_id].item()
    
    # Return 1 for positive, 0 for negative (matching Yelp label format)
    return 1 if prob_pos > prob_neg else 0


def parse_llama_response(response, original_text):
    if "Output:" in response:
        output_part = response.split("Output:")[-1].strip()
    else:
        output_part = response.strip()

    words = []
    if ',' in output_part:
        words = [word.strip() for word in output_part.split(',')]
    else:
        words = re.findall(r'\b[\w\'-]+\b', output_part)

    text_lower = original_text.lower()
    valid_words = [word for word in words
                   if word.lower() in text_lower
                   and len(word) > 2] 

    seen = set()
    unique_words = []
    for word in valid_words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)

    return unique_words[:3]

# Step 5: Collect correct predictions with short sentences
correct_short_samples = []
print("Evaluating model on test data...")

# We limit the loop for demonstration, remove slice [:1000] for full run
for sample in tqdm(test_data, desc="Processing test samples"):
    text = sample['sentence']
    label = sample['label']  # 0 for negative, 1 for positive
    
    pred = predict_sentiment(model, tokenizer, text)
    
    if pred == label:  # Correct prediction
        word_count = len(text.split())
        if 5 <= word_count <= 10:  # Filter for short sentences
            correct_short_samples.append({'text': text, 'label': label})
            
    # Optional: Stop early if you have enough samples
    #if len(correct_short_samples) >= 50:
        #break

print(f"Found {len(correct_short_samples)} correctly predicted short samples.")

# Step 6: Function to query LLaMA 3 
hf_token = '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' 
llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=hf_token)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def query_llama3(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that extracts sentiment words from reviews."},
                {"role": "user", "content": prompt}
            ]
            input_ids = llama_tokenizer.apply_chat_template(messages, return_tensors="pt").to(llama_model.device)
            
            outputs = llama_model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=llama_tokenizer.eos_token_id,
                eos_token_id=llama_tokenizer.eos_token_id,
                num_return_sequences=1
            )
            response = llama_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            if response and any(char.isalpha() for char in response):
                return response
        except Exception as e:
            print(f"Query attempt {attempt + 1} failed: {str(e)}")
    return ""

# Step 7: Identify sensitive words using LLaMA 3
sensitive_words = []
print("Identifying sensitive words with LLaMA 3...")

PROMPT_TEMPLATE = (
    "Given that this review has a {sentiment} sentiment, identify up to three words that most strongly indicate this {sentiment} sentiment. "
    "Only select words that appear in the review text. Return the words in a comma-separated list. "
    "If no words strongly indicate the sentiment, return an empty list.\n\n"
    "Examples:\n"
    "Review: The food was amazing and the service was excellent!\n"
    "Sentiment: positive\n"
    "Output: amazing, excellent\n"
    "Review: Terrible experience, never coming back.\n"
    "Sentiment: negative\n"
    "Output: terrible\n"
    "Review: It was okay, nothing special.\n"
    "Sentiment: negative\n"
    "Output: \n\n"
    "Review: {text}\n"
    "Sentiment: {sentiment}\n"
    "Output: "
)

for sample in tqdm(correct_short_samples, desc="Querying LLaMA 3"):
    text = sample['text']
    label = sample['label']
    sentiment = "positive" if label == 1 else "negative"
    prompt = PROMPT_TEMPLATE.format(sentiment=sentiment, text=text)

    response = query_llama3(prompt)
    valid_words = parse_llama_response(response, text) if response else []
    
    sensitive_words.append({
        'text': text,
        'label': sentiment,
        'sensitive_word': valid_words
    })

# Step 8: Save the results to a file
output_file = "/users/sglli24/UnderstandingFineTuningViaMI/src/find_corrupt_data/gpt2_sst2_sensitive_words.json"
# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(sensitive_words, f, indent=4)
print(f"Results saved to '{output_file}'.")

# Step 9: Display a few samples for manual verification
print("\nSample results for manual verification:")
for i in range(min(5, len(sensitive_words))):
    print(f"Sample {i+1}:")
    print(f"Text: {sensitive_words[i]['text']}")
    print(f"Label: {sensitive_words[i]['label']}")
    print(f"Sensitive Word: {sensitive_words[i]['sensitive_word']}")
    print()