import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

print("[*] Locating cached model weights...")
file_path = hf_hub_download(repo_id="lerobot/pi0_libero_base", filename="model.safetensors")

print("[*] Searching for trapped normalization statistics...")
weights = load_file(file_path)

mean_key = None
std_key = None

# Dynamically hunt for the correct keys
for key in weights.keys():
    if "action.mean" in key and ("unnormalize" in key or "normalize" in key):
        mean_key = key
    elif "action.std" in key and ("unnormalize" in key or "normalize" in key):
        std_key = key

if mean_key and std_key:
    action_mean = weights[mean_key].numpy()
    action_std = weights[std_key].numpy()
    print("\n=== ACTION STATISTICS FOUND ===")
    print(f"Exact Mean Key: {mean_key}")
    print(f"Exact STD Key:  {std_key}")
    print(f"MEAN: {list(action_mean)}")
    print(f"STD:  {list(action_std)}")
else:
    print("\n[!] Could not find the exact keys. Dumping all possible matches:")
    for key in weights.keys():
        if "action" in key and ("mean" in key or "std" in key):
            print(f" - {key}")