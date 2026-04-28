import argparse
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import visualize_dataset

# 1. Konfiguracja parametrów wejściowych (argparse)
parser = argparse.ArgumentParser(description="Wizualizacja datasetu LeRobot z wymuszonym backendem PyAV")
parser.add_argument(
    "--episode", 
    "-e", 
    type=int, 
    default=0, 
    help="Indeks epizodu, który chcesz wyświetlić (domyślnie: 0)"
)
args = parser.parse_args()

print(f"[*] Ładowanie datasetu dla epizodu: {args.episode}...")

# 2. Inicjalizacja datasetu
dataset = LeRobotDataset(
    repo_id="local/panda_pick_and_place",
    root="/home/student/bartosz_niedzielski/panda/magisterka/camera/outputs/panda_pick_and_place"
)

# 3. Wstrzyknięcie backendu PyAV (rozwiązanie problemu z bibliotekami)
dataset.video_backend = "pyav"
print("[*] Backend PyAV został pomyślnie wymuszony!")

# 4. Uruchomienie wizualizatora dla konkretnego epizodu
# Funkcja visualize_dataset oczekuje listy indeksów, dlatego przekazujemy [args.episode]
visualize_dataset(
    dataset=dataset, 
    episode_index=[args.episode], 
    mode="local"
)