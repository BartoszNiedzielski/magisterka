from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 1. Define your paths and names
local_folder_path = "outputs/hard_pick_and_place" 
target_repo_id = "bartek-niedzielski/hard_pick_and_place_70" 

print(f"Loading local dataset from {local_folder_path}...")

# 2. Load the dataset locally using its internal local ID
dataset = LeRobotDataset(
    "local/hard_pick_and_place", 
    root=local_folder_path
)

print(f"Dataset loaded! Contains {dataset.num_episodes} episodes.")

# --- THE CRITICAL FIX ---
# Forcefully change the dataset's internal identity to your Hugging Face username
dataset.repo_id = target_repo_id 
# ------------------------

print(f"Pushing to Hugging Face Hub as '{dataset.repo_id}'...")

# 3. Push to the Hub
dataset.push_to_hub(
    private=False,            
    tags=["lerobot", "panda"] 
)

print("✅ Upload complete!")