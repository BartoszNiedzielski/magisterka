import json
import torch
from pathlib import Path

# Handle LeRobot's recent folder restructuring
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.compute_stats import compute_stats
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.compute_stats import compute_stats

print("[*] Loading local dataset...")
dataset_dir = Path("outputs/panda_pick_task")
dataset = LeRobotDataset("local/panda_pick_task", root=dataset_dir)

print("[*] Computing statistics (this may take a minute depending on dataset size)...")
# batch_size=8 prevents RAM overload when reading video frames
stats = compute_stats(dataset, batch_size=8, num_workers=4)

print("[*] Saving stats.json to disk...")
# Format tensors to standard lists so they can be saved as JSON
stats_dict = {}
for key, stat in stats.items():
    stats_dict[key] = {
        k: v.tolist() if hasattr(v, 'tolist') else v 
        for k, v in stat.items()
    }

# Save directly into the LeRobot meta folder
stats_file = dataset_dir / "meta" / "stats.json"
with open(stats_file, "w") as f:
    json.dump(stats_dict, f, indent=2)

print(f"✅ Success! Stats computed and saved to {stats_file}")