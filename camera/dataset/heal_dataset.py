import os
import glob
import json
import pandas as pd

dataset_path = "/home/student/bartosz_niedzielski/panda/magisterka/camera/outputs/panda_pick_task"
meta_dir = os.path.join(dataset_path, "meta")

print("[*] Analyzing dataset corruption...")

# 1. Get the ground truth of valid episodes from the metadata we already cleaned
valid_episodes = set()
episodes_file = os.path.join(meta_dir, "episodes.jsonl")

if not os.path.exists(episodes_file):
    print("[!] Cannot find episodes.jsonl!")
    exit()

with open(episodes_file, 'r') as f:
    for line in f:
        valid_episodes.add(json.loads(line)["episode_index"])

print(f"[*] Valid episodes according to metadata: {sorted(list(valid_episodes))}")
print("[*] Scrubbing all Parquet subfolders to remove any ghost data...\n")

# 2. Fix the Glob to search subfolders recursively (This was the missing link!)
chunk_files = glob.glob(os.path.join(dataset_path, "data", "**", "*.parquet"), recursive=True)

if not chunk_files:
    print("[!] No parquet files found! Check the dataset path.")

for file_path in chunk_files:
    # Read the parquet file
    df = pd.read_parquet(file_path)
    initial_rows = len(df)
    
    # Only keep rows that belong to the strictly valid episodes
    df_clean = df[df['episode_index'].isin(valid_episodes)]
    
    # Overwrite the file if we found ghost data
    if len(df_clean) < initial_rows:
        df_clean.to_parquet(file_path)
        print(f"[*] Cleaned {os.path.basename(file_path)}: Surgically removed {initial_rows - len(df_clean)} ghost frames.")
    else:
        print(f"[*] {os.path.basename(file_path)} is already clean.")

print("\n[+] Dataset healed! The Parquet files now perfectly match the metadata.")