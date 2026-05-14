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

# Camera Indices
EXTERIOR_CAMERA_INDEX = 2
WRIST_CAMERA_INDEX = 0

logging.basicConfig(level=logging.INFO)

# --- Shared Variables for Threading ---
is_recording = False
trajectory_buffer = []
current_grip_state = 0.0 # 0.0 = Open, 1.0 = Closed

episode_successful = True

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
    gripper.move(width=0.08, speed=0.1)
    gripper.homing()
    
    # --- DROID-LeRobot v2.1 Schema ---
    dataset_dir = Path("outputs/panda_pick_and_place")
    
    if dataset_dir.exists():
        print("[*] Found existing dataset. Loading it to append a new episode...")
        dataset = LeRobotDataset("local/panda_pick_and_place", root=dataset_dir)
    else:
        print("[*] Creating new dataset folder...")
        dataset = LeRobotDataset.create(
            repo_id="local/panda_pick_and_place",
            root=dataset_dir,
            features={
                "observation/exterior_image_1_left": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
                "observation/wrist_image_left": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
                "observation/joint_position": {"dtype": "float32", "shape": (7,), "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]},
                "observation/gripper_position": {"dtype": "float32", "shape": (1,), "names": ["grip"]},
                "actions": {"dtype": "float32", "shape": (8,), "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]},
            },
            fps=FPS,
            image_writer_threads=4,
        )

    print("[*] Homing robot and opening gripper...")
    panda.move_to_start(speed_factor=0.2)
    gripper.move(width=0.08, speed=0.1)
    pose = panda.get_pose()
    pose[2,3] -= 0.1 
    q = panda_py.ik(pose)
    panda.move_to_joint_position(q, speed_factor=0.2)
    time.sleep(1)
    current_grip_state = 0.0

    # --- 2. Teaching Phase ---
    print('\n--- Teaching Mode: Poses ---')
    positions = []
    panda.teaching_mode(True)

    for i in range(1):
        input(f'Manually move the arm to Pose {i+1} and press Enter...')
        positions.append(panda.q)
        pose = panda.get_pose()
        print(pose)
        print(panda.get_position())

    panda.teaching_mode(False) 

    # --- Prep for Replay ---
    input('\nPress Enter to move to Start Position (Pose 1)... and replay the trajectory while recording data.')
    panda.move_to_start(speed_factor=0.2)
    gripper.move(width=0.08, speed=0.1)
    pose = panda.get_pose()
    pose[2,3] -= 0.1 
    q = panda_py.ik(pose)
    panda.move_to_joint_position(q, speed_factor=0.2)
    time.sleep(2)

    # --- 4. Replay & Record Phase ---
    print("RECORDING STARTED...")
    trajectory_buffer = []
    is_recording = True
    
    rec_thread = threading.Thread(target=recording_thread, args=(panda, cap_ext, cap_wrist))
    rec_thread.start()

    try:
        print("Moving to Position 1...")
        panda.move_to_joint_position(positions[0], speed_factor=0.1)

        print("Grasping...")
        gripper.grasp(width=0.0, speed=0.1, force=40.0)
        current_grip_state = 1.0 

        pose = panda.get_pose()
        pose[2,3] += 0.1
        panda.move_to_pose(pose,speed_factor=0.1)

        print("move to prep drop area")
        drop_pose = np.array([
        [ 0.996302, -0.07637459, 0.03911541, 0.61723842],
        [-0.0788481, -0.99467823, 0.06617418, 0.16486688],
        [ 0.03385322, -0.06901365, -0.9970411, 0.12573585],
        [ 0.0,          0.0,          0.0,          1.0        ]
        ], dtype=np.float64)
        q = panda_py.ik(drop_pose)
        panda.move_to_joint_position(q, speed_factor=0.2)

        print("moving to place area")
        place_pose = np.array([
            [ 0.99673478, -0.07589372,  0.02721537,  0.60739891],
            [-0.07610894, -0.99706556,  0.00695978,  0.16566014],
            [ 0.02660730, -0.00900838, -0.99960536,  0.03852027],
            [ 0.0,         0.0,         0.0,         1.0       ]
        ], dtype=np.float64)
        q = panda_py.ik(place_pose)
        panda.move_to_joint_position(q, speed_factor=0.1)

        gripper.move(width=0.08, speed=0.1)
        current_grip_state = 0.0

        pose = panda.get_pose()
        pose[2,3] += 0.1
        panda.move_to_pose(pose,speed_factor=0.1)

    except Exception as e:
        print(f"\n[!] ERROR DETECTED DURING REPLAY: {e}")
        print("[!] Skipping dataset save for this corrupted episode.")
        episode_successful = False
        
        try:
            panda.recover()
        except:
            pass

    finally:
        # D. Stop Recording Safely
        is_recording = False
        rec_thread.join()
        
        cap_ext.release()
        cap_wrist.release()
        print("RECORDING STOPPED.")

    if episode_successful:

        # --- 5. Format and Save to LeRobot ---
        print(f"\nProcessing {len(trajectory_buffer)} frames for LeRobot...")
        instruction = "place the green cube in the yellow area"

        for i in range(len(trajectory_buffer) - 1):
            current_state_q, current_grip, ext_img, wrist_img = trajectory_buffer[i]
            next_state_q, next_grip, _, _ = trajectory_buffer[i + 1]
            
            # Action vector is now absolute joint positions, not velocities
            action_vector = np.concatenate([next_state_q, [next_grip]]).astype(np.float32)

            frame = {
                "observation/exterior_image_1_left": ext_img,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": current_state_q,
                "observation/gripper_position": np.array([current_grip], dtype=np.float32),
                "actions": action_vector,
                "task": instruction
            }
            
            dataset.add_frame(frame)

        dataset.save_episode()
        print("Dataset episode saved successfully!")

 # data visualization sanity check (optional)
    # command to visualize:
    # python -m lerobot.scripts.visualize_dataset   --repo-id local/panda_pick_and_place   --root /home/student/bartosz_niedzielski/panda/magisterka/camera/outputs/panda_pick_and_place   --mode local   --episode-index 13
