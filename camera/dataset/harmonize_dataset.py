import json
from pathlib import Path

dataset_path = Path("outputs/panda_pick_task/meta/info.json")

if dataset_path.exists():
    with open(dataset_path, "r") as f:
        info = json.load(f)
    
    # Check if 'action' exists and 'actions' doesn't
    features = info.get("features", {})
    if "action" in features and "actions" not in features:
        print("[*] Harmonizing: Renaming 'action' to 'actions' in metadata...")
        features["actions"] = features.pop("action")
        
        with open(dataset_path, "w") as f:
            json.dump(info, f, indent=2)
        print("✅ Metadata updated!")
    else:
        print("[-] Metadata already contains 'actions' or 'action' is missing.")
else:
    print("❌ Could not find info.json. Check your path!")