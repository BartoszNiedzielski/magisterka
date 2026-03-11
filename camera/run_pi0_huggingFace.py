# deactivating CUDA to force bfloat16 CPU inference for testing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import cv2
import numpy as np
import torch
import threading
import logging
from scipy.spatial.transform import Rotation as R

import panda_py
from panda_py import libfranka

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame

import queue

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'
INSTRUCTION = "pick up the green cube"

CAM_INDEX_3RD_PERSON = 2
CAM_INDEX_WRIST = 0

logging.basicConfig(level=logging.INFO)

# === SHARED VARIABLES ===
action_queue = queue.Queue(maxsize=20)
state_lock = threading.Lock()
latest_chunk = None         
chunk_timestamp = 0.0       
is_running = True

# Global robot instances
desk = None
panda = None
gripper = None

# === ACTION NORMALIZATION STATISTICS ===
ACTION_MEAN = np.array([0.062781565, 0.08684081, -0.09037306, 0.00054074306, 0.00564338, -0.0052290987, -0.049640723], dtype=np.float32)
ACTION_STD = np.array([0.33552372, 0.378447, 0.4447286, 0.03924354, 0.06339297, 0.077970274, 0.99876714], dtype=np.float32)
STATE_MEAN = np.array([-0.046518784, 0.03440907, 0.7645525, 2.9722095, -0.22046979, -0.1255794, 0.026914254, -0.027190784], dtype=np.float32)
STATE_STD = np.array([0.10494395, 0.1517662, 0.3785167, 0.34427345, 0.90694684, 0.3253919, 0.014175904, 0.0140588945], dtype=np.float32)


# ==========================================
# THREAD 1: THE BRAIN (Hugging Face bfloat16 Pi0)
# ==========================================
def vision_loop():
    global latest_chunk, chunk_timestamp, is_running, panda
    
    print("[Brain] Booting Dual Cameras...")
    cap_3rd = cv2.VideoCapture(CAM_INDEX_3RD_PERSON)
    cap_wrist = cv2.VideoCapture(CAM_INDEX_WRIST)
    
    if not cap_3rd.isOpened() or not cap_wrist.isOpened():
        print("[Brain] ERROR: Could not open cameras.")
        is_running = False
        return
    
    print("[Brain] Loading Pi0 in Native bfloat16 Precision...")
    device = "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/pi0_libero_base"

    print("test")

    try:
    # 1. Load directly to device and cast weights to bfloat16
        policy = PI0Policy.from_pretrained(
            model_id,
            device=device,
            low_cpu_mem_usage=True
        )
        # policy = PI0Policy.from_pretrained(
        #     model_id, 
        #     torch_dtype=torch.bfloat16, 
        #     low_cpu_mem_usage=True
        # ).to(device)
        policy.eval()
    except Exception as e:
        print(f"[Brain] ERROR loading Pi0 model: {e}")
        is_running = False
        return
    print("test_2")

    # 2. Get the official LeRobot processors
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    print("[Brain] bfloat16 Pi0 Online. Listening for visual updates...")
    
    chunk_id = 0
    step_in_chunk = 0

    try:
        while is_running:
            # 3. Read the frames
            ret_3rd, frame_3rd = cap_3rd.read()
            ret_wrist, frame_wrist = cap_wrist.read()
            
            if not ret_3rd or not ret_wrist:
                print("[Brain] Missed camera frame, skipping...")
                continue

            vis_3rd = frame_3rd.copy()
            vis_wrist = frame_wrist.copy()

            # --- THE VISUAL BLINDFOLD FIX ---
            # 1. Convert OpenCV BGR to PyTorch RGB
            frame_3rd_rgb = cv2.cvtColor(frame_3rd, cv2.COLOR_BGR2RGB)
            frame_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)

            # 2. Resize to exactly 224x224 (as dictated by the pi0 config)
            frame_3rd_rgb = cv2.resize(frame_3rd_rgb, (224, 224))
            frame_wrist_rgb = cv2.resize(frame_wrist_rgb, (224, 224))

            # 3. Transpose to (Channels, Height, Width) AND scale pixels to [0.0, 1.0]
            model_3rd = np.transpose(frame_3rd_rgb, (2, 0, 1)).astype(np.float32) / 255.0
            model_wrist = np.transpose(frame_wrist_rgb, (2, 0, 1)).astype(np.float32) / 255.0
            
            # Dummy camera also needs to be Channels-First and float32
            dummy_camera = np.zeros((3, 224, 224), dtype=np.float32)

            # 4. Get Robot State Safely
            try:
                current_pose = np.array(panda.get_pose()).reshape(4, 4)
                pos = current_pose[:3, 3]
                euler = R.from_matrix(current_pose[:3, :3]).as_euler('xyz')
                
                # Create the 8-dimension state: [X, Y, Z, Roll, Pitch, Yaw, Finger1, Finger2]
                state_8d = np.concatenate([pos, euler, [0.0, 0.0]]).astype(np.float32)
                
                # Normalize the physical state so the AI understands it
                state_np = (state_8d - STATE_MEAN) / STATE_STD

            except Exception as e:
                print(f"[Brain] Error reading panda state: {e}")
                continue

            # 5. Build standard LeRobot Inference Frame
            obs_dict = {
                "observation.state": torch.from_numpy(state_np),
                "observation.images.image": torch.from_numpy(model_3rd),
                "observation.images.image2": torch.from_numpy(model_wrist),
                "observation.images.empty_camera_0": torch.from_numpy(dummy_camera),
                "task": INSTRUCTION
            }

            start_time = time.time()
            # 6. Autocast to ensure our inputs map correctly to the bf16 model
            with torch.inference_mode():
                processed_obs = preprocess(obs_dict)
                action_chunk = policy.select_action(processed_obs)
                processed_action = postprocess(action_chunk)

            interference_time = time.time() - start_time
            action_np = processed_action.cpu().numpy()[0]
            
            # ... (the rest of your queue logic remains exactly the same) ...

            if interference_time > 0.02:
                chunk_id += 1
                step_in_chunk = 0
            else:
                step_in_chunk += 1

            tracked_action = (chunk_id, step_in_chunk, action_np)

            if action_queue.full():
                action_queue.get()  # Discard oldest if we're at capacity

            action_queue.put(tracked_action)
            
            # # Send to muscle thread
            # with state_lock:
            #     latest_chunk = processed_action.cpu().numpy()[0] 
            #     # print("latest chunk:", latest_chunk)
            #     # print("latest_chunk shape:", latest_chunk.shape)
            #     chunk_timestamp = time.time()  

            # visualization
            debug_view = np.hstack((cv2.resize(vis_3rd, (320, 240)), cv2.resize(vis_wrist, (320, 240))))
            cv2.putText(debug_view, f"Latency: {time.time() - start_time:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Pi0 Vision", debug_view)
            
            if cv2.waitKey(1) == ord('q'):
                is_running = False
                break
                
    finally:
        cap_3rd.release()
        cap_wrist.release()
        cv2.destroyAllWindows()
        print("[Brain] Shutting down.")


# ==========================================
# THREAD 2: THE MUSCLE (Robot Control)
# ==========================================
def control_loop():
    global is_running, panda, gripper
    
    print("[Muscle] Robot Online. Ready to move.")

    current_chunk_id = -1
    
    # We will store the offset between the virtual world and the real world here
    frame_offset_pos = np.zeros(3)
    frame_offset_rot = R.identity()

    gripper_is_closed = False

    while is_running:
        try:
            chunk_id, step_in_chunk, current_action = action_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # 1. Un-normalize the model's output to get Virtual Absolute Coordinates
        current_action = (current_action * ACTION_STD) + ACTION_MEAN
        target_x, target_y, target_z, target_roll, target_pitch, target_yaw, target_grip = current_action[:7]

        # Convert virtual target to numpy array and scipy Rotation object
        virtual_pos = np.array([target_x, target_y, target_z])
        virtual_rot = R.from_euler('xyz', [target_roll, target_pitch, target_yaw], degrees=False)

        # 2. ANCHOR THE TRAJECTORY (Calculate the Offset on Step 0)
        if chunk_id != current_chunk_id:
            print(f"\n[Muscle] >>> ANCHORING NEW TRAJECTORY CHUNK: #{chunk_id} <<<")
            current_chunk_id = chunk_id
            
            # Find out where the robot is physically sitting RIGHT NOW
            current_pose = np.array(panda.get_pose()).reshape(4, 4)
            physical_pos = current_pose[:3, 3]
            physical_rot = R.from_matrix(current_pose[:3, :3])
            
            # Calculate the mathematical difference between Real Life and the Virtual World
            frame_offset_pos = physical_pos - virtual_pos
            frame_offset_rot = physical_rot * virtual_rot.inv()

        # 3. Apply the Offset to map the virtual absolute point into our real room
        target_pos = virtual_pos + frame_offset_pos
        target_rot = frame_offset_rot * virtual_rot

        # 4. NEW SAFETY CLAMPS (Limit Step Speed, Not Absolute Position)
        current_pose = np.array(panda.get_pose()).reshape(4, 4)
        current_pos = current_pose[:3, 3]
        
        # Calculate how far this specific step wants us to move from our current physical spot
        step_delta_pos = target_pos - current_pos
        print(f"[Muscle] Raw Target Pos: {target_pos}, Current Pos: {current_pos}, Step Delta: {step_delta_pos}")
        
        # Clamp the physical delta to a maximum of 5cm per 20ms to prevent violent jerks
        step_delta_pos = np.clip(step_delta_pos, -0.05, 0.05) 
        safe_target_pos = current_pos + step_delta_pos

        if (safe_target_pos[0] > 0.7 or safe_target_pos[0] < 0 
            or safe_target_pos[1] > 0.3 or safe_target_pos[1] < -0.3 
            or safe_target_pos[2] > 0.65 or safe_target_pos[2] < 0.05): 
            print("Warning: Attempted to move beyond safe box limits.")
            continue

        print(f"[Muscle] Chunk {chunk_id} Step {step_in_chunk} | Target X: {safe_target_pos[0]:.3f}, Y: {safe_target_pos[1]:.3f}, Z: {safe_target_pos[2]:.3f} | Grip: {target_grip:.2f}")

        # 5. EXECUTE PHYSICAL MOVEMENT
        target_pose_mat = np.eye(4)
        target_pose_mat[:3, :3] = target_rot.as_matrix()
        target_pose_mat[:3, 3]  = safe_target_pos
        
        try:
            panda.move_to_pose(target_pose_mat, speed_factor=0.05)
            
            should_close = target_grip < 0.0 
            
            # Only send the command if the model wants to change the current state
            if should_close and not gripper_is_closed:
                print("[Muscle] -> Executing GRASP")
                gripper.grasp(width=0.0, speed=0.1, force=40)
                gripper_is_closed = True
                
            elif not should_close and gripper_is_closed:
                print("[Muscle] -> Executing RELEASE")
                gripper.move(width=0.08, speed=0.1)
                gripper_is_closed = False
        except Exception as e:
            print(f"[Muscle] Motion Error: {e}")
        
        # Wait 20ms before grabbing the next action from the queue
        time.sleep(0.02) 

    print("[Muscle] Shutting down.")


# ==========================================
# MAIN ROUTINE
# ==========================================
if __name__ == "__main__":
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
        pose[2,3] -= 0.35
        q = panda_py.ik(pose)
        panda.move_to_joint_position(q, speed_factor=0.05)
        gripper.move(width=0.08, speed=0.1)
        time.sleep(1)
    except Exception as e:
        print(f"[*] Fatal Error connecting to Robot: {e}")
        exit(1)

    print("[*] Booting bfloat16 Pi0 Asynchronous Framework...")
    brain_thread = threading.Thread(target=vision_loop)
    muscle_thread = threading.Thread(target=control_loop)
    
    brain_thread.start()
    muscle_thread.start()
    
    try:
        brain_thread.join()
        muscle_thread.join()
    except KeyboardInterrupt:
        is_running = False
        
    print("[*] Releasing Robot Control...")
    try:
        desk.lock()
        desk.release_control()
    except Exception:
        pass
        
    print("[*] System safely powered down.")