from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

print("[*] Locating cached model weights...")
file_path = hf_hub_download(repo_id="lerobot/pi0_libero_base", filename="model.safetensors")
weights = load_file(file_path)

mean_key, std_key = None, None
for key in weights.keys():
    if "observation_state.mean" in key and "normalize" in key:
        mean_key = key
    elif "observation_state.std" in key and "normalize" in key:
        std_key = key

if mean_key and std_key:
    print(f"STATE_MEAN = np.array({list(weights[mean_key].numpy())}, dtype=np.float32)")
    print(f"STATE_STD = np.array({list(weights[std_key].numpy())}, dtype=np.float32)")