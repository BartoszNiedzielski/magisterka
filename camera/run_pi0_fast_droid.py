import time
import cv2
import numpy as np
import threading
import logging

# --- OpenPI Imports ---
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

import panda_py
from panda_py import libfranka
# from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'

INSTRUCTION = "pick up the green cube"
ACTION_SCALE = 1
CONTROL_HZ = 20  # How fast the muscle thread executes actions (20Hz = 0.05s per step)

# --- CAMERA CONFIGURATION ---
EXTERIOR_CAMERA_INDEX = 2
WRIST_CAMERA_INDEX = 0

logging.basicConfig(level=logging.INFO)

# === SHARED VARIABLES ===
state_lock = threading.Lock()
latest_action_chunk = None 
is_running = True

# Initialize globals so threads know they exist
gripper = None
panda = None
desk = None

# ==========================================
# THREAD 1: THE BRAIN (Vision & AI)
# ==========================================
def vision_loop():
    global latest_action_chunk, is_running, gripper, panda
    
    print(f"[Brain] Starting Logitech Cameras (Exterior: {EXTERIOR_CAMERA_INDEX}, Wrist: {WRIST_CAMERA_INDEX})...")
    
    cap_ext = cv2.VideoCapture(EXTERIOR_CAMERA_INDEX)
    cap_wrist = cv2.VideoCapture(WRIST_CAMERA_INDEX)

    if not cap_ext.isOpened():
        print(f"[Brain] ERROR: Could not open Exterior Camera {EXTERIOR_CAMERA_INDEX}.")
        is_running = False
        return
    if not cap_wrist.isOpened():
        print(f"[Brain] ERROR: Could not open Wrist Camera {WRIST_CAMERA_INDEX}.")
        is_running = False
        return

    print("[Brain] Downloading/Loading Pi0 Droid model...")
    pi0_config = _config.get_config("pi05_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    policy = policy_config.create_trained_policy(pi0_config, checkpoint_dir)

    print("[Brain] AI Online. Listening for visual updates...")
    
    try:
        while is_running:
            ret_ext, frame_ext = cap_ext.read()
            ret_wrist, frame_wrist = cap_wrist.read()
            
            if not ret_ext or not ret_wrist: 
                continue
            
            image_ext_rgb = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
            image_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)

            # Safely grab the current grip state and joint positions directly from the hardware
            with state_lock:
                is_closed = gripper.read_once().is_grasped
                current_grip_state = 1.0 if is_closed else 0.0
                # Fetch the 7 joint angles from the robot
                current_joints = np.array(panda.q, dtype=np.float32)

            # The updated dictionary with ALL required DROID parameters
            example = {
                "observation/exterior_image_1_left": image_ext_rgb,
                "observation/wrist_image_left": image_wrist_rgb,
                "observation/gripper_position": np.array([current_grip_state], dtype=np.float32),
                "observation/joint_position": current_joints,
                "prompt": INSTRUCTION
            }

            start_time = time.time()
            
            # Run Inference
            result = policy.infer(example)
            action_chunk = result["actions"] 
            
            # Convert the entire chunk to a numpy array
            if hasattr(action_chunk, 'cpu'): action_chunk = action_chunk.cpu().numpy()
            elif hasattr(action_chunk, 'device'): action_chunk = np.array(action_chunk)

            
            print(f"[Brain] Generated chunk of {len(action_chunk)} actions. (Latency: {time.time() - start_time:.3f}s)")

            h_ext, w_ext, _ = image_ext_rgb.shape
            h_wrist, w_wrist, _ = image_wrist_rgb.shape

            # Resize wrist to match external height (keeping aspect ratio or forcing match)
            image_wrist_resized = cv2.resize(image_wrist_rgb, (int(w_wrist * h_ext / h_wrist), h_ext))

            # 2. Stack them horizontally (External on Left, Wrist on Right)
            combined_view = np.hstack((image_ext_rgb, image_wrist_resized))

            # 3. Convert back to BGR for OpenCV display (imshow expects BGR)
            display_frame = cv2.cvtColor(combined_view, cv2.COLOR_RGB2BGR)

            try:
                cv2.imshow("Robot Cameras (Left: Ext | Right: Wrist)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Add your break logic here
                    pass
            except cv2.error:
                # FALLBACK: If imshow still fails, save to file so you can at least see it
                cv2.imwrite("camera_debug_view.jpg", display_frame)
                print("[!] GUI Error: Could not open window. Frame saved to 'camera_debug_view.jpg'")

            # Hand the ENTIRE chunk over to the muscle thread
            with state_lock:
                latest_action_chunk = action_chunk.copy()
                
    finally:
        cap_ext.release()
        cap_wrist.release()
        print("[Brain] Shutting down.")


# ==========================================
# THREAD 2: THE MUSCLE (Robot Control)
# ==========================================
def control_loop():
    global latest_action_chunk, is_running, panda, gripper

    current_chunk = None
    step_index = 0
    time_per_step = 1.0 / CONTROL_HZ

    while is_running:
        step_start = time.time()

        # 1. Check if the Brain has given us a new trajectory
        with state_lock:
            if latest_action_chunk is not None:
                current_chunk = latest_action_chunk
                latest_action_chunk = None  # Clear it so we don't grab it again
                step_index = 0              # Reset to the start of the new chunk

        # 2. If we have a chunk to execute, process the next step
        if current_chunk is not None and step_index < len(current_chunk):
            action = current_chunk[step_index]
            step_index += 1

            # Extract the 7 joint velocities and the 1 gripper state
            target_velocities = action[:7]
            grip = action[7]

            with state_lock:
                current_joints = np.array(panda.q, dtype=np.float32)
            
            # Physics 101: Position = Current Position + (Velocity * Time)
            # This calculates exactly where the 7 joints should be at the end of this 0.05s step
            target_joints = current_joints + (target_velocities * time_per_step)

            try:
                # Calculate where these joint angles WILL put the hand
                predicted_pose = np.array(panda_py.fk(target_joints))
                safe_target_pos = predicted_pose[:3, 3] # Extract X, Y, Z
                
                # Your physical bounding box check
                if (safe_target_pos[0] > 0.7 or safe_target_pos[0] < 0 
                    or safe_target_pos[1] > 0.3 or safe_target_pos[1] < -0.3 
                    or safe_target_pos[2] > 0.65 or safe_target_pos[2] < 0.05): 
                    print(f"[Warning] Movement rejected. Predicted POS: {safe_target_pos}")
                    continue  # Skip moving to stay safe!
                
            except Exception as e:
                print(f"[Muscle] FK calculation failed: {e}")
                continue

            # We can skip the XYZ bounding box check here because we are in joint space,
            # but Franka's internal limits will automatically stop it if it bends too far.

            # Send the new joint angles directly to the robot
            panda.move_to_joint_position(target_joints, speed_factor=0.02)
            
            try:
                # DROID: >= 0.5 means CLOSED
                if grip >= 0.5:
                    gripper.grasp(width=0.0, speed=0.1, force=40)
                    with state_lock: latest_grip = 1.0
                else:
                    gripper.move(width=0.08, speed=0.1)
                    with state_lock: latest_grip = 0.0
            except Exception:
                pass

        # 3. Sleep just enough to maintain our control frequency
        elapsed = time.time() - step_start
        sleep_time = max(0, time_per_step - elapsed)
        time.sleep(sleep_time)
    
    print("[Muscle] Shutting down.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[*] Booting Asynchronous Robotics System...")
    print("[*] Initializing Robot Connection First...")
    try:
        desk = panda_py.Desk(ROBOT_IP, ROBOT_USER, ROBOT_PASS)
        desk.unlock()
        desk.activate_fci()

        panda = panda_py.Panda(ROBOT_IP)
        gripper = libfranka.Gripper(ROBOT_IP)
        
        print("[*] Homing robot...")
        panda.move_to_start(speed_factor=0.05)
        pose = panda.get_pose()
        pose[2,3] -= 0.3
        q = panda_py.ik(pose)
        panda.move_to_joint_position(q, speed_factor=0.05)
        gripper.move(width=0.08, speed=0.1)
        time.sleep(1)
    except Exception as e:
        print(f"[*] Fatal Error connecting to Robot: {e}")
        exit(1)
    
    brain_thread = threading.Thread(target=vision_loop)
    muscle_thread = threading.Thread(target=control_loop)
    
    brain_thread.start()
    muscle_thread.start()
    
    # --- NEW SHUTDOWN LOGIC ---
    try:
        # Loop with a tiny timeout so the main thread can "listen" for Ctrl+C
        while brain_thread.is_alive() and muscle_thread.is_alive():
            brain_thread.join(timeout=0.1)
            muscle_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        print("\n[*] Ctrl+C detected! Signaling threads to shut down safely...")
        is_running = False  # This breaks the 'while is_running' loops in both threads!
    
    # Safely wait for threads to finish their current loop iteration
    brain_thread.join()
    muscle_thread.join()
    
    try:
        print("[*] Locking brakes and releasing FCI...")
        desk.lock()
        desk.release_control()
    except Exception:
        pass
    
    print("[*] System safely powered down.")