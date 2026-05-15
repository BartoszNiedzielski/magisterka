from huggingface_hub import snapshot_download

repo_id = "bartek-niedzielski/hard_pick_and_place_25"
local_directory = "outputs/hard_pick_and_place"

print(f"Downloading {repo_id}...")

# This downloads all files (parquet, videos, json) into your target folder
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_directory,
    local_dir_use_symlinks=False # Forces actual file downloads, not just shortcuts
)

print(f"✅ Download complete! Files saved to {local_directory}")