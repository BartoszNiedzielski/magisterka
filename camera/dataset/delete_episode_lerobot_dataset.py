import os
import glob
import json
import pandas as pd

# --- Configuration ---
dataset_path = "/home/student/bartosz_niedzielski/panda/magisterka/camera/outputs/panda_pick_and_place"
meta_dir = os.path.join(dataset_path, "meta")

print("[*] Scanning dataset to find the last episode...")

# 1. Safely find the highest episode_index currently in the dataset
episodes_file = os.path.join(meta_dir, "episodes.jsonl")
if not os.path.exists(episodes_file):
    print("[!] Error: episodes.jsonl not found. Is the dataset empty?")
    exit()

max_episode_index = -1
with open(episodes_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        idx = data.get("episode_index", -1)
        if idx > max_episode_index:
            max_episode_index = idx

if max_episode_index == -1:
    print("[!] No episodes found to delete.")
    exit()

episodes_to_delete = [max_episode_index]
print(f"[*] Found it! Surgically removing Episode {max_episode_index}...\n")

# 2. Clean the Parquet Data Chunks
# CRITICAL: We use recursive=True so it looks inside the chunk-000 subfolders!
chunk_files = glob.glob(os.path.join(dataset_path, "data", "**", "*.parquet"), recursive=True)
for file_path in chunk_files:
    df = pd.read_parquet(file_path)
    initial_rows = len(df)
    
    # Keep only rows that are NOT the episode we are deleting
    df_clean = df[~df['episode_index'].isin(episodes_to_delete)]
    
    # Overwrite the file if we actually removed something
    if len(df_clean) < initial_rows:
        # If the dataframe is now completely empty, delete the file so HF doesn't crash!
        if len(df_clean) == 0:
            os.remove(file_path)
            print(f"[*] Deleted empty shell file: {os.path.basename(file_path)}")
        else:
            df_clean.to_parquet(file_path)
            print(f"[*] Cleaned {os.path.basename(file_path)}: Dropped {initial_rows - len(df_clean)} frames.")

# Clean episodes.parquet (if it exists)
ep_parquet = os.path.join(meta_dir, "episodes.parquet")
if os.path.exists(ep_parquet):
    df_ep = pd.read_parquet(ep_parquet)
    df_ep = df_ep[~df_ep['episode_index'].isin(episodes_to_delete)]
    df_ep.to_parquet(ep_parquet)

# 3. Clean the JSONL Metadata
frames_removed = 0
for filename in ["episodes.jsonl", "episodes_stats.jsonl"]:
    filepath = os.path.join(meta_dir, filename)
    if not os.path.exists(filepath): 
        continue
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    valid_lines = []
    for line in lines:
        data = json.loads(line)
        if data.get("episode_index") in episodes_to_delete:
            if filename == "episodes.jsonl":
                frames_removed += data.get("length", 0)
        else:
            valid_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(valid_lines)
    print(f"[*] Cleaned {filename}")

# 4. Update info.json math
info_path = os.path.join(meta_dir, "info.json")
if os.path.exists(info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    info["total_episodes"] = max(0, info.get("total_episodes", 0) - 1)
    if "total_frames" in info:
        info["total_frames"] = max(0, info["total_frames"] - frames_removed)
        
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"[*] Updated info.json (Subtracted {frames_removed} frames).")

# 5. Locate and Delete the Video Files
videos_dir = os.path.join(dataset_path, "videos")
if os.path.exists(videos_dir):
    # LeRobot pads episode numbers with 6 zeros (e.g., 000015)
    padded_index = f"{max_episode_index:06d}"
    
    # Search all subfolders in the videos directory for the MP4s
    video_files = glob.glob(os.path.join(videos_dir, "**", f"*_{padded_index}.mp4"), recursive=True)
    video_files.extend(glob.glob(os.path.join(videos_dir, "**", f"*episode_{max_episode_index}.mp4"), recursive=True))
    
    # Remove duplicates
    video_files = list(set(video_files))
    
    for vid_path in video_files:
        try:
            os.remove(vid_path)
            print(f"[*] Deleted video: {os.path.basename(vid_path)}")
        except Exception as e:
            print(f"[!] Could not delete video {vid_path}: {e}")

print(f"\n[+] Success! Episode {max_episode_index} and its videos have been completely erased from the dataset.")