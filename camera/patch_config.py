from huggingface_hub import snapshot_download
import json
import os

print("[*] Downloading Pi0 model locally (This will take a few minutes for the heavy weights)...")
local_dir = "./pi0_local"
snapshot_download(repo_id="lerobot/pi0", local_dir=local_dir)

config_path = os.path.join(local_dir, "config.json")

print("[*] Opening config.json to scrub outdated keys...")
with open(config_path, "r") as f:
    data = json.load(f)

bad_keys = [
    "resize_imgs_with_padding", "adapt_to_pi_aloha", "use_delta_joint_actions_aloha",
    "proj_width", "num_steps", "use_cache", "attention_implementation", "train_state_proj"
]

for key in bad_keys:
    if key in data:
        print(f" -> Removing ghost key: {key}")
        del data[key]

with open(config_path, "w") as f:
    json.dump(data, f, indent=4)

print("[*] Patch complete! The model is ready to load.")