import logging
import time
import threading
import numpy as np
import cv2
from pathlib import Path
import panda_py
from panda_py import libfranka
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
hostname = '172.16.0.2'
username = 'Dentec'
password = 'Frankenstein'
FPS = 15

# Camera Indices (Adjust these based on your OS assignment)
EXTERIOR_CAMERA_INDEX = 0
WRIST_CAMERA_INDEX = 2

logging.basicConfig(level=logging.INFO)

# --- Shared Variables for Threading ---
is_recording = False
trajectory_buffer = []
current_grip_state = 0.0 # 0.0 = Open, 1.0 = Closed

def recording_thread(panda, cap_ext, cap_wrist):
    """Background loop that captures state AND cameras exactly at 15Hz"""
    global is_recording, trajectory_buffer, current_grip_state
    
    while is_recording:
        step_start = time.time()
        
        # 1. Capture Camera Frames
        ret_ext, frame_ext = cap_ext.read()
        ret_wrist, frame_wrist = cap_wrist.read()
        
        if not ret_ext or not ret_wrist:
            logging.warning("[!] Missed a camera frame!")
            continue

        # Convert OpenCV BGR format to standard RGB for the AI model
        rgb_ext = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
        rgb_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
        
        # 2. Capture Robot State
        current_q = np.array(panda.q, dtype=np.float32)
        
        # 3. Store in memory buffer
        trajectory_buffer.append((current_q, current_grip_state, rgb_ext, rgb_wrist))
        
        # 4. Maintain strict 15Hz frequency
        elapsed = time.time() - step_start
        time.sleep(max(0, (1.0 / FPS) - elapsed))

if __name__ == '__main__':
    # --- 1. Initialization ---
    print("[*] Initializing Cameras...")
    cap_ext = cv2.VideoCapture(EXTERIOR_CAMERA_INDEX)
    cap_wrist = cv2.VideoCapture(WRIST_CAMERA_INDEX)
    
    # Force camera resolution to standard 480x640 to match Droid config
    for cap in (cap_ext, cap_wrist):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap_ext.isOpened() or not cap_wrist.isOpened():
        raise RuntimeError("Failed to open one or both cameras. Check indices.")

    print("[*] Connecting to Robot...")
    desk = panda_py.Desk(hostname, username, password)
    desk.unlock()
    desk.activate_fci()

    panda = panda_py.Panda(hostname)
    gripper = libfranka.Gripper(hostname)
    gripper.homing()
    
    # Setup LeRobot Dataset schema to include cameras and 8D states/actions
    dataset_dir = Path("outputs/panda_pick_task")
    
    if dataset_dir.exists():
        print("[*] Found existing dataset. Loading it to append a new episode...")
        # Initialize normally to append to the existing dataset
        dataset = LeRobotDataset("local/panda_pick_task", root=dataset_dir)
    else:
        print("[*] Creating new dataset folder...")
        # Create fresh dataset schema
        dataset = LeRobotDataset.create(
            repo_id="local/panda_pick_task",
            root=dataset_dir,
            features={
                "observation.images.exterior": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
                "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
                "observation.state": {"dtype": "float32", "shape": (8,), "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]},
                "actions": {"dtype": "float32", "shape": (8,), "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]},
            },
            fps=FPS,
            image_writer_threads=4,
        )

    print("[*] Homing robot and opening gripper...")
    panda.move_to_start(speed_factor=0.1)
    gripper.move(width=0.08, speed=0.1)
    pose = panda.get_pose()
    pose[2,3] -= 0.1 #see if it is good
    q = panda_py.ik(pose)
    panda.move_to_joint_position(q, speed_factor=0.1)
    gripper.move(width=0.08, speed=0.1)
    time.sleep(1)
    current_grip_state = 0.0

    # --- 2. Teaching Phase ---
    print('\n--- Teaching Mode: Poses ---')
    positions = []
    panda.teaching_mode(True) 

    for i in range(2):
        input(f'Manually move the arm to Pose {i+1} and press Enter...')
        positions.append(panda.q)

    panda.teaching_mode(False) 

    # --- Prep for Replay ---
    input('\nPress Enter to move to Start Position (Pose 1)...')
    panda.move_to_start(speed_factor=0.1)
    gripper.move(width=0.08, speed=0.1)
    pose = panda.get_pose()
    pose[2,3] -= 0.1 #see if it is good
    q = panda_py.ik(pose)
    panda.move_to_joint_position(q, speed_factor=0.1)
    gripper.move(width=0.08, speed=0.1)
    time.sleep(1)

    input('\nReady to record. Press Enter to Replay and Save to Dataset...')
    # --- 4. Replay & Record Phase ---
    print("🔴 RECORDING STARTED...")
    trajectory_buffer = []
    is_recording = True
    
    # Start the background watcher, passing the camera objects
    rec_thread = threading.Thread(target=recording_thread, args=(panda, cap_ext, cap_wrist))
    rec_thread.start()

    try:
        print("Moving to Position 1...")
        panda.move_to_joint_position(positions[0], speed_factor=0.1)

        print("Grasping...")
        gripper.grasp(width=0.0, speed=0.1, force=40.0)
        current_grip_state = 1.0 # Update state so the recording thread sees it

        print("Moving to Position 2...")
        panda.move_to_joint_position(positions[1], speed_factor=0.1)

    finally:
        # D. Stop Recording Safely
        is_recording = False
        rec_thread.join()
        
        # Release cameras
        cap_ext.release()
        cap_wrist.release()
        print("⏹️ RECORDING STOPPED.")

    # --- 5. Format and Save to LeRobot ---
    print(f"\nProcessing {len(trajectory_buffer)} frames for LeRobot...")
    
    # Define the string once to ensure no typos
    instruction = "pick up the green cube"

    for i in range(len(trajectory_buffer) - 1):
        current_state_q, current_grip, ext_img, wrist_img = trajectory_buffer[i]
        next_state_q, next_grip, _, _ = trajectory_buffer[i + 1]
        
        state_vector = np.concatenate([current_state_q, [current_grip]]).astype(np.float32)
        action_vector = np.concatenate([next_state_q, [next_grip]]).astype(np.float32)

        # Create the frame dictionary
        frame = {
            "observation.images.exterior": ext_img,
            "observation.images.wrist": wrist_img,
            "observation.state": state_vector,
            "actions": action_vector,
            "task": instruction
        }
        
        dataset.add_frame(frame)

    dataset.save_episode()
    print("✅ Dataset episode saved successfully!")