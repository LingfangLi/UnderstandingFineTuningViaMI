import torch


model_path = "/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/qwen2-coqa.pt"

print(f"Loading: {model_path} ...")
state_dict = torch.load(model_path, map_location="cpu")

keys = list(state_dict.keys())
print(f"\nTotal keys in .pt file: {len(keys)}")

print("\nTop 10 keys in .pt file:")
for k in keys[:10]:
    print(k)

print("\nChecking key patterns:")

if any("blocks." in k for k in keys):
    print("Detected 'blocks.' -> Likely TransformerLens format")

elif any("lora" in k for k in keys):
    print("Detected 'lora' -> Likely Adapter weights")

elif any("module." in k for k in keys):
    print("Detected 'module.' -> Likely DDP wrapped")
else:
    print("Unknown format. Please copy the Top 10 keys above.")