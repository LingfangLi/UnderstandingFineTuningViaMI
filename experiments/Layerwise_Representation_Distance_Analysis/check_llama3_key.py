import torch
import os

pt_path = r"/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/llama3.2-kde4.pt"

print(f"Inspecting file: {pt_path}")
state_dict = torch.load(pt_path, map_location="cpu")

print("\n=== Top 5 Keys in your .pt file ===")
for i, k in enumerate(list(state_dict.keys())[:5]):
    print(f"{i}: {k}")

print("\n=== Comparison Check ===")
has_std = "model.embed_tokens.weight" in state_dict
print(f"Contains 'model.embed_tokens.weight'? : {has_std}")

has_prefix = any("base_model" in k for k in state_dict.keys())
print(f"Keys contain 'base_model'? : {has_prefix}")
