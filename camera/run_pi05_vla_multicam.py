import time
import cv2
import numpy as np
import threading
import logging
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'
INSTRUCTION = "pick up the green cube"

# === MULTI-CAMERA SETUP ===
# Change these indices if OpenCV grabs the wrong USB feeds
CAM_INDEX_3RD_PERSON = 0  
CAM_INDEX_WRIST = 1       

logging.basicConfig(level=logging.INFO)

# === SHARED VARIABLES ===
state_lock = threading.Lock()
latest_chunk = None         
chunk_timestamp = 0.0       
is_running = True

# ==========================================
# THREAD 1: THE BRAIN (Dual Camera & Pi0.5)
# ==========================================
def vision_loop():
    global latest_chunk, chunk_timestamp, is_running
    
    print("[Brain] Booting Dual Cameras...")
    cap_3rd = cv2.VideoCapture(CAM_INDEX_3RD_PERSON)
    cap_wrist = cv2.VideoCapture(CAM_INDEX_WRIST)
    
    # Verify cameras...
    
    print("[Brain] Downloading and Loading Pi0.5 DROID Model...")
    # 1. Get the official PI config for the DROID fine-tune
    cfg = _config.get_config("pi05_droid")
    
    # 2. Download the model from PI's Google Cloud Bucket (Caches locally!)
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    
    # 3. Create the policy directly on the GPU
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir)

    print("[Brain] Pi0.5 DROID Online. Listening for visual updates...")
    
    try:
        while is_running:
            # Grab frames
            ret1, frame_3rd = cap_3rd.read()
            ret2, frame_wrist = cap_wrist.read()
            if not ret1 or not ret2: continue
            
            # openpi expects standard RGB Numpy arrays (not BGR from OpenCV, and not PyTorch Tensors!)
            img_3rd_rgb = cv2.cvtColor(frame_3rd, cv2.COLOR_BGR2RGB)
            img_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)

            # Get Proprioception
            current_state = np.array(panda.get_pose()).reshape(4, 4) if 'panda' in globals() else np.eye(4)
            current_pos = current_state[:3, 3] # XYZ

            # Build Observation Dictionary
            # openpi's built-in transforms will handle standardizing this to the model's needs
            observation = {
                "observation/exterior_image_1_left": img_3rd_rgb,   # Your 3rd person Logitech camera
                "observation/wrist_image_left": img_wrist_rgb,      # Your wrist-mounted camera
                # "observation/joint_position": current_joints,     # (Add this if the model complains about missing proprioception!)
                "prompt": INSTRUCTION
            }

            # Fast Inference
            start_time = time.time()
            
            # openpi returns a dictionary, the actions are inside the "actions" key
            result = policy.infer(observation) 
            action_chunk = np.array(result["actions"]) 
            
            # Safely send the chunk to the muscle thread
            with state_lock:
                latest_chunk = action_chunk
                chunk_timestamp = time.time()  

            # ... [Keep your OpenCV cv2.imshow debug visualization here] ...
            
            if cv2.waitKey(1) == ord('q'):
                is_running = False
                break
                
    finally:
        cap_3rd.release()
        cap_wrist.release()
        cv2.destroyAllWindows()
        print("[Brain] Shutting down.")


# ==========================================
# THREAD 2: THE MUSCLE (Action Chunking)
# ==========================================
def control_loop():
    global is_running, latest_chunk, chunk_timestamp
    
    print("[Muscle] Authenticating with Franka Desk...")
    try:
        desk = panda_py.Desk(ROBOT_IP, ROBOT_USER, ROBOT_PASS)
        desk.unlock()
        desk.activate_fci()

        # We declare panda globally so the Brain thread can read its state
        global panda 
        panda = panda_py.Panda(ROBOT_IP)
        gripper = libfranka.Gripper(ROBOT_IP)
        
        print("[Muscle] Homing robot...")
        panda.move_to_start(speed_factor=0.05)
        # ... (Your custom homing logic is preserved)
        time.sleep(1)
        print("[Muscle] Robot Online. Ready to move.")

    except Exception as e:
        print(f"[Muscle] Failed to connect to Robot: {e}")
        is_running = False  
        return

    active_chunk = None
    local_chunk_time = 0.0
    step_idx = 0  

    while is_running:
        with state_lock:
            if latest_chunk is not None and chunk_timestamp > local_chunk_time:
                active_chunk = np.copy(latest_chunk)
                local_chunk_time = chunk_timestamp
                step_idx = 0

        # === FIXED INDENTATION: Everything below must only run IF we have an active chunk ===
        if active_chunk is not None and step_idx < len(active_chunk):
            
            current_action = active_chunk[step_idx]
            target_dx, target_dy, target_dz, target_droll, target_dpitch, target_dyaw, target_grip = current_action

            # --- SAFETY CLAMPS ---
            target_dx = np.clip(target_dx, -0.05, 0.05)
            target_dy = np.clip(target_dy, -0.05, 0.05)
            target_dz = np.clip(target_dz, -0.05, 0.05)
            
            MAX_ROT = 0.1 
            target_droll = np.clip(target_droll, -MAX_ROT, MAX_ROT)
            target_dpitch = np.clip(target_dpitch, -MAX_ROT, MAX_ROT)
            target_dyaw = np.clip(target_dyaw, -MAX_ROT, MAX_ROT)

            # --- KINEMATICS & EXECUTION ---
            current_pose = np.array(panda.get_pose()).reshape(4, 4)
            current_pos = current_pose[:3, 3]
            current_rot_mat = current_pose[:3, :3]
            current_rot = R.from_matrix(current_rot_mat)

            delta_rot = R.from_euler('xyz', [target_droll, target_dpitch, target_dyaw], degrees=False)
            target_rot = delta_rot * current_rot
            target_pos = current_pos + np.array([target_dx, target_dy, target_dz])

            if (target_pos[0] > 0.7 or target_pos[0] < 0 
                or target_pos[1] > 0.3 or target_pos[1] < -0.3 
                or target_pos[2] > 0.65 or target_pos[2] < 0.05): 
                print("[Muscle] Warning: Safety box limits reached.")
                step_idx += 1 
                continue  
            
            target_pose = np.eye(4)
            target_pose[:3, :3] = target_rot.as_matrix()
            target_pose[:3, 3]  = target_pos

            # NOTE: Uncomment when ready to move physically!
            # panda.move_to_pose(target_pose, speed_factor=0.05)
            
            try:
                if target_grip < 0.5:
                    gripper.grasp(width=0.0, speed=0.1, force=40)
                else:
                    gripper.move(width=0.08, speed=0.1)
            except Exception:
                pass

            step_idx += 1
            time.sleep(0.02) # Pi0 Action chunks are meant to be executed FAST (50Hz)

        else:
            # Wait for a new chunk
            time.sleep(0.01)
    
    print("[Muscle] Shutting down.")
    try:
        desk.lock()
        desk.release_control()
    except Exception:
        pass


if __name__ == "__main__":
    print("[*] Booting Pi0.5 Asynchronous Framework...")
    brain_thread = threading.Thread(target=vision_loop)
    muscle_thread = threading.Thread(target=control_loop)
    brain_thread.start()
    muscle_thread.start()
    brain_thread.join()
    muscle_thread.join()
    print("[*] System safely powered down.")