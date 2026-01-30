import torch


MODEL_PATH = r"/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/llama3.2-kde4.pt"

print(f"Inspecting: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location="cpu")


if 'blocks.0.attn.W_Q' in state_dict:
    weight = state_dict['blocks.0.attn.W_Q']
    print(f"\ Found Key: blocks.0.attn.W_Q")
    print(f" Shape: {weight.shape}")


    d_model = weight.shape[1]

    print(f"\n>>> DETECTED d_model: {d_model}")

    if d_model == 2048:
        print("VERDICT: This is a 1B Model.")
        print("   (Llama-3.2-1B has hidden size 2048)")
        print("   6GB size likely due to Float32 (full precision) storage.")
    elif d_model == 3072:
        print("VERDICT: This is a 3B Model.")
        print("   (Llama-3.2-3B has hidden size 3072)")
    else:
        print(f"Unknown Config. d_model={d_model}")

else:
    print("Key not found. Something is very weird.")
