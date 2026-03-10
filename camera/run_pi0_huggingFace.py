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

            # Resize just in case
            frame_3rd = cv2.resize(frame_3rd, (256, 256))
            frame_wrist = cv2.resize(frame_wrist, (256, 256))

            dummy_camera = np.zeros((224, 224, 3), dtype=np.uint8)

            frame_3rd = np.transpose(frame_3rd, (2, 0, 1))
            frame_wrist = np.transpose(frame_wrist, (2, 0, 1))
            dummy_camera = np.transpose(dummy_camera, (2, 0, 1))

            # 4. Get Robot State Safely
            try:
                # Get 4x4 transform matrix from libfranka
                print(panda.get_state())
                current_pose = np.array(panda.get_pose()).reshape(4, 4)
                pos = current_pose[:3, 3]
                euler = R.from_matrix(current_pose[:3, :3]).as_euler('xyz')
                state_6d = np.concatenate([pos, euler]) 
                state_np = np.append(state_6d, 0.0).astype(np.float32)
            except Exception as e:
                print(f"[Brain] Error reading panda state: {e}")
                continue

            # 5. Build standard LeRobot Inference Frame
            obs_dict = {
                "observation.state": torch.from_numpy(state_np),
                "observation.images.image": torch.from_numpy(frame_3rd),
                "observation.images.image2": torch.from_numpy(frame_wrist),
                "observation.images.empty_camera_0": torch.from_numpy(dummy_camera),
                "task": INSTRUCTION
            }

            start_time = time.time()
            # 6. Autocast to ensure our inputs map correctly to the bf16 model
            with torch.inference_mode():
                processed_obs = preprocess(obs_dict)
                action_chunk = policy.select_action(processed_obs)
                processed_action = postprocess(action_chunk)
                print("processed_action shape:", processed_action.shape)
                print("processed action", processed_action)
            # with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            #     processed_obs = preprocess(obs_dict)
            #     action_chunk = policy.select_action(processed_obs)
            #     processed_action = postprocess(action_chunk)

            interference_time = time.time() - start_time
            action_np = processed_action.cpu().numpy()[0]

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
    global is_running, latest_chunk, chunk_timestamp, panda, gripper
    
    print("[Muscle] Robot Online. Ready to move.")

    active_chunk = None
    local_chunk_time = 0.0
    step_idx = 0
    current_chunk_id = -1

    while is_running:

        try:
            chunk_id, step_in_chunk, current_action = action_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        # with state_lock:
        #     if latest_chunk is not None and chunk_timestamp > local_chunk_time:
        #         active_chunk = np.copy(latest_chunk)
                
        #         # --- THE FIX: FORCE 2D SHAPE ---
        #         if active_chunk.ndim == 1:
        #             active_chunk = np.expand_dims(active_chunk, axis=0)
                    
        #         local_chunk_time = chunk_timestamp
        #         step_idx = 0
        #         # print("new chunk received")

        # if active_chunk is not None and step_idx < len(active_chunk):
        #     current_action = active_chunk[step_idx]

        # Detect if this is the start of a new trajectory
        if chunk_id != current_chunk_id:
            print(f"\n[Muscle] >>> STARTING NEW TRAJECTORY CHUNK: #{chunk_id} <<<")
            current_chunk_id = chunk_id

        print("current_action before denorm:", current_action)
        print("current_action shape:", current_action.shape)

        current_action = (current_action * ACTION_STD) + ACTION_MEAN
        
        target_dx, target_dy, target_dz, target_droll, target_dpitch, target_dyaw, target_grip = current_action[:7]

        print(f"[Muscle] Chunk {chunk_id} Step {step_in_chunk} | dX={target_dx:.3f}, dY={target_dy:.3f}, dZ={target_dz:.3f} | Grip={target_grip:.2f}")
        # print(f"[Muscle] Step {step_idx}: dX={target_dx:.3f}, dY={target_dy:.3f}, dZ={target_dz:.3f} | Grip={target_grip:.2f}")

        # --- SAFETY CLAMPS ---
        target_dx = np.clip(target_dx, -0.05, 0.05)
        target_dy = np.clip(target_dy, -0.05, 0.05)
        target_dz = np.clip(target_dz, -0.05, 0.05)
        
        MAX_ROT = 0.1 
        target_droll = np.clip(target_droll, -MAX_ROT, MAX_ROT)
        target_dpitch = np.clip(target_dpitch, -MAX_ROT, MAX_ROT)
        target_dyaw = np.clip(target_dyaw, -MAX_ROT, MAX_ROT)

        current_pose = np.array(panda.get_pose()).reshape(4, 4)
        current_pos = current_pose[:3, 3]
        current_rot_mat = current_pose[:3, :3]
        current_rot = R.from_matrix(current_rot_mat)

        delta_rot = R.from_euler('xyz', [target_droll, target_dpitch, target_dyaw], degrees=False)
        target_rot = delta_rot * current_rot
        target_pos = current_pos + np.array([target_dx, target_dy, target_dz])

        # Un-comment this block to actually move the physical robot!
        '''
        target_pose = np.eye(4)
        target_pose[:3, :3] = target_rot.as_matrix()
        target_pose[:3, 3]  = target_pos
        try:
            panda.move_to_pose(target_pose, speed_factor=0.05)
            
            if target_grip < 0.5:
                gripper.grasp(width=0.0, speed=0.1, force=40)
            else:
                gripper.move(width=0.08, speed=0.1)
        except Exception as e:
            print(f"[Muscle] Motion Error: {e}")
        '''
        step_idx += 1
        time.sleep(0.02) 

        # else:
        #     time.sleep(0.01)
    
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
        pose[2,3] -= 0.2
        q = panda_py.ik(pose)
        panda.move_to_joint_position(q, speed_factor=0.05)
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