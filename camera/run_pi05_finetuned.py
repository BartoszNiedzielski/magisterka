import time
import cv2
import numpy as np
import threading
import logging

# --- OpenPI Imports ---
from openpi.training import config as _config
from openpi.policies import policy_config

import panda_py
from panda_py import libfranka

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'

INSTRUCTION = "place the green cube in the yellow area"
CHECKPOINT_DIR = "/home/student/ft/checkpoints/pi05-panda-pos-v1/checkpoint_250"
CONTROL_HZ = 15  # How fast the muscle thread executes actions (15Hz = 0.067s per step)

# --- CAMERA CONFIGURATION ---
EXTERIOR_CAMERA_INDEX = 2
WRIST_CAMERA_INDEX = 0

logging.basicConfig(level=logging.INFO)

# === SHARED VARIABLES ===
state_lock = threading.Lock()
latest_action_chunk = None 
is_running = True
latest_grip = 0.0

# Initialize globals
gripper = None
panda = None
desk = None

# ==========================================
# THREAD 1: THE BRAIN (Vision & AI)
# ==========================================
def vision_loop(cap_ext, cap_wrist, policy):
    global latest_action_chunk, is_running, gripper, panda
    print("[Brain] AI Online. Listening for visual updates...")
    
    try:
        while is_running:
            ret_ext, frame_ext = cap_ext.read()
            ret_wrist, frame_wrist = cap_wrist.read()
            
            if not ret_ext or not ret_wrist: 
                continue
            
            image_ext_rgb = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
            image_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)

            with state_lock:
                is_closed = gripper.read_once().is_grasped
                current_grip_state = 1.0 if is_closed else 0.0
                current_joints = np.array(panda.q, dtype=np.float32)

            example = {
                "observation/exterior_image_1_left": image_ext_rgb,
                "observation/wrist_image_left": image_wrist_rgb,
                "observation/gripper_position": np.array([current_grip_state], dtype=np.float32),
                "observation/joint_position": current_joints,
                "prompt": INSTRUCTION
            }

            result = policy.infer(example)
            action_chunk = result["actions"] 
            
            if hasattr(action_chunk, 'cpu'): action_chunk = action_chunk.cpu().numpy()
            elif hasattr(action_chunk, 'device'): action_chunk = np.array(action_chunk)

            h_ext, w_ext, _ = image_ext_rgb.shape
            h_wrist, w_wrist, _ = image_wrist_rgb.shape

            image_wrist_resized = cv2.resize(image_wrist_rgb, (int(w_wrist * h_ext / h_wrist), h_ext))
            combined_view = np.hstack((image_ext_rgb, image_wrist_resized))
            display_frame = cv2.cvtColor(combined_view, cv2.COLOR_RGB2BGR)

            try:
                cv2.imshow("Robot Cameras (Left: Ext | Right: Wrist)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            except cv2.error:
                pass # Fail silently if display isn't available

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
    global latest_action_chunk, is_running, panda, gripper, latest_grip

    current_chunk = None
    step_index = 0
    time_per_step = 1.0 / CONTROL_HZ

    while is_running:
        step_start = time.time()

        with state_lock:
            if latest_action_chunk is not None:
                current_chunk = latest_action_chunk
                latest_action_chunk = None  
                step_index = 0
                print(f"[Muscle] Received new action chunk with {len(current_chunk)} steps.")       

        if current_chunk is not None and step_index < len(current_chunk):
            action = current_chunk[step_index]
            step_index += 1
            print(f"action: {action}")

            # Extract absolute joint positions directly from the action array
            target_joints = action[:7].astype(np.float64)
            grip = action[7]

            try:
                # Validate the target position using Forward Kinematics
                predicted_pose = np.array(panda_py.fk(target_joints))
                safe_target_pos = predicted_pose[:3, 3] 
                
                if (safe_target_pos[0] > 0.7 or safe_target_pos[0] < 0 
                    or safe_target_pos[1] > 0.3 or safe_target_pos[1] < -0.3 
                    or safe_target_pos[2] > 0.65 or safe_target_pos[2] < 0.06): 
                    
                    print(f"[Warning] Movement rejected. Predicted POS: {safe_target_pos}")
                    # Skip this step to avoid collision
                    continue
                # else:
                #     print(f"Predicted POS: {safe_target_pos}")
                
            except Exception as e:
                print(f"[Muscle] FK calculation failed: {e}")
                continue

            # Execute the safe joint position target using native method
            try:
                # Using a very fast speed factor to allow the 15Hz loop to dictate the pace
                panda.move_to_joint_position(target_joints, speed_factor=0.05)
                print(f"[Muscle] Step {step_index}/{len(current_chunk)} | Target Pos Executed | Grip: {grip:.2f}")
            except Exception as e:
                print(f"[Muscle] Execution skipped/failed: {e}")
            
            try:
                if grip >= 0.5:
                    print("should be closing")
                with state_lock:
                    current_grip_state = latest_grip

                if grip >= 0.5 and current_grip_state == 0.0:
                    print("[Muscle] Closing gripper...")
                    gripper.grasp(width=0.0, speed=0.1, force=40)
                    with state_lock: latest_grip = 1.0
                elif grip < 0.5 and current_grip_state == 1.0:
                    print("[Muscle] Opening gripper...")
                    gripper.move(width=0.08, speed=0.1)
                    with state_lock: latest_grip = 0.0
            except Exception:
                pass

        elapsed = time.time() - step_start
        sleep_time = max(0, time_per_step - elapsed)
        time.sleep(sleep_time)
    
    print("[Muscle] Shutting down.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[*] Booting Asynchronous Robotics System...")
    
    # --- 1. INITIALIZE AI & CAMERAS FIRST ---
    print(f"[*] Starting Logitech Cameras (Exterior: {EXTERIOR_CAMERA_INDEX}, Wrist: {WRIST_CAMERA_INDEX})...")
    cap_ext = cv2.VideoCapture(EXTERIOR_CAMERA_INDEX)
    cap_wrist = cv2.VideoCapture(WRIST_CAMERA_INDEX)

    if not cap_ext.isOpened() or not cap_wrist.isOpened():
        print("[!] ERROR: Could not open cameras. Check indices.")
        exit(1)

    print(f"[*] Loading fine-tuned Pi0 model from: {CHECKPOINT_DIR} ...")
    pi0_config = _config.get_config("pi05_panda_pos_finetune") 
    policy = policy_config.create_trained_policy(pi0_config, CHECKPOINT_DIR)

    # --- 2. THE WARM-UP PASS ---
    print("\n[*] Warming up the AI (This will take ~20 seconds)...")
    ret_ext, frame_ext = cap_ext.read()
    ret_wrist, frame_wrist = cap_wrist.read()
    
    if ret_ext and ret_wrist:
        image_ext_rgb = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
        image_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
        
        dummy_joints = np.zeros(7, dtype=np.float32)
        dummy_grip = np.zeros(1, dtype=np.float32)

        warmup_example = {
            "observation/exterior_image_1_left": image_ext_rgb,
            "observation/wrist_image_left": image_wrist_rgb,
            "observation/gripper_position": dummy_grip,
            "observation/joint_position": dummy_joints,
            "prompt": INSTRUCTION
        }
        
        start_time = time.time()
        _ = policy.infer(warmup_example)
        print(f"[+] AI Warmup Complete! (Took {time.time() - start_time:.2f}s)\n")
    else:
        print("[!] ERROR: Could not read cameras for warmup.")
        exit(1)

    # --- 3. CONNECT TO ROBOT ---
    print("[*] Initializing Robot Connection...")
    try:
        desk = panda_py.Desk(ROBOT_IP, ROBOT_USER, ROBOT_PASS)
        desk.unlock()
        desk.activate_fci()

        panda = panda_py.Panda(ROBOT_IP)
        gripper = libfranka.Gripper(ROBOT_IP)
        
        print("[*] Homing robot...")
        panda.move_to_start(speed_factor=0.05)
        pose = panda.get_pose()
        pose[2,3] -= 0.1
        q = panda_py.ik(pose)
        panda.move_to_joint_position(q, speed_factor=0.05)
        gripper.move(width=0.08, speed=0.1)
        time.sleep(1)

    except Exception as e:
        print(f"[*] Fatal Error connecting to Robot: {e}")
        exit(1)
    
    # --- 4. START THREADS ---
    brain_thread = threading.Thread(target=vision_loop, args=(cap_ext, cap_wrist, policy))
    muscle_thread = threading.Thread(target=control_loop)
    
    brain_thread.start()
    muscle_thread.start()
    
    try:
        while brain_thread.is_alive() and muscle_thread.is_alive():
            brain_thread.join(timeout=0.1)
            muscle_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        print("\n[*] Ctrl+C detected! Signaling threads to shut down safely...")
        is_running = False 
    
    brain_thread.join()
    muscle_thread.join()
    
    try:
        print("[*] Locking brakes...")
        desk.lock()
        desk.release_control()
    except Exception:
        pass
    
    print("[*] System safely powered down.")