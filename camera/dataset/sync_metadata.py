import os
import json

dataset_path = "/home/student/bartosz_niedzielski/panda/magisterka/camera/outputs/panda_pick_task"
meta_dir = os.path.join(dataset_path, "meta")

print("[*] Auditing metadata against physical Parquet files...")

episodes_file = os.path.join(meta_dir, "episodes.jsonl")
if not os.path.exists(episodes_file):
    print("[!] No metadata found.")
    exit()

valid_lines_ep = []
missing_episodes = []
frames_removed = 0

# 1. Read the Table of Contents
with open(episodes_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)
    ep_idx = data["episode_index"]
    
    # 2. Check if the actual Parquet file exists on the hard drive
    padded_idx = f"{ep_idx:06d}"
    expected_file = os.path.join(dataset_path, "data", "chunk-000", f"episode_{padded_idx}.parquet")
    
    if os.path.exists(expected_file):
        valid_lines_ep.append(line)
    else:
        print(f"[!] Missing Parquet for Episode {ep_idx}. Scrubbing from metadata...")
        missing_episodes.append(ep_idx)
        frames_removed += data.get("length", 0)

if not missing_episodes:
    print("\n[+] Metadata is already perfectly synced!")
else:
    # 3. Rewrite episodes.jsonl
    with open(episodes_file, 'w') as f:
        f.writelines(valid_lines_ep)
        
    # 4. Rewrite episodes_stats.jsonl
    stats_file = os.path.join(meta_dir, "episodes_stats.jsonl")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats_lines = f.readlines()
        valid_stats = [l for l in stats_lines if json.loads(l)["episode_index"] not in missing_episodes]
        with open(stats_file, 'w') as f:
            f.writelines(valid_stats)
            
    # 5. Rewrite info.json
    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        info["total_episodes"] = max(0, info.get("total_episodes", 0) - len(missing_episodes))
        if "total_frames" in info:
            info["total_frames"] = max(0, info["total_frames"] - frames_removed)
            
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
            
    print(f"\n[+] Success! Scrubbed {len(missing_episodes)} ghost episodes from the metadata.")