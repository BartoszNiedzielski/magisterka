# GPU Inference Activated!
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import time
import cv2
import numpy as np
import torch
import threading
import logging
import queue
from scipy.spatial.transform import Rotation as R

import panda_py
from panda_py import libfranka

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

logging.basicConfig(level=logging.INFO)

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'
INSTRUCTION = "pick up the green block"

CAM_INDEX_3RD_PERSON = 2
CAM_INDEX_WRIST = 0

# === SHARED VARIABLES ===
action_queue = queue.Queue(maxsize=20)
is_running = True

desk = None
panda = None
gripper = None

# === NORMALIZATION STATISTICS ===
ACTION_MEAN = np.array([0.062781565, 0.08684081, -0.09037306, 0.00054074306, 0.00564338, -0.0052290987, -0.049640723], dtype=np.float32)
ACTION_STD = np.array([0.33552372, 0.378447, 0.4447286, 0.03924354, 0.06339297, 0.077970274, 0.99876714], dtype=np.float32)
STATE_MEAN = np.array([-0.046518784, 0.03440907, 0.7645525, 2.9722095, -0.22046979, -0.1255794, 0.026914254, -0.027190784], dtype=np.float32)
STATE_STD = np.array([0.10494395, 0.1517662, 0.3785167, 0.34427345, 0.90694684, 0.3253919, 0.014175904, 0.0140588945], dtype=np.float32)


# ==========================================
# THREAD 1: THE BRAIN (Hugging Face Pi0)
# ==========================================
def vision_loop():
    global is_running, panda, gripper
    
    print("[Brain] Booting Dual Cameras...")
    cap_3rd = cv2.VideoCapture(CAM_INDEX_3RD_PERSON)
    cap_wrist = cv2.VideoCapture(CAM_INDEX_WRIST)
    
    if not cap_3rd.isOpened() or not cap_wrist.isOpened():
        print("[Brain] ERROR: Could not open cameras.")
        is_running = False
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Brain] Loading Pi0 on {device}...")
    model_id = "lerobot/pi0_libero_base"

    try:
        # strict=False saves the model from amnesia!
        policy = PI0Policy.from_pretrained(
            model_id,
            device=device,
            low_cpu_mem_usage=True,
            strict=False
        )
        policy.eval()
    except Exception as e:
        print(f"[Brain] ERROR loading Pi0 model: {e}")
        is_running = False
        return

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    print("[Brain] Pi0 Online. Listening for visual updates...")
    
    chunk_id = 0
    step_in_chunk = 0

    try:
        while is_running:
            ret_3rd, frame_3rd = cap_3rd.read()
            ret_wrist, frame_wrist = cap_wrist.read()
            
            if not ret_3rd or not ret_wrist:
                continue

            vis_3rd = frame_3rd.copy()
            vis_wrist = frame_wrist.copy()

            # --- VISUAL PREPARATION ---
            frame_3rd_rgb = cv2.cvtColor(frame_3rd, cv2.COLOR_BGR2RGB)
            frame_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)

            frame_3rd_rgb = cv2.resize(frame_3rd_rgb, (224, 224))
            frame_wrist_rgb = cv2.resize(frame_wrist_rgb, (224, 224))

            model_3rd = np.transpose(frame_3rd_rgb, (2, 0, 1)).astype(np.float32) / 255.0
            model_wrist = np.transpose(frame_wrist_rgb, (2, 0, 1)).astype(np.float32) / 255.0
            dummy_camera = np.zeros((3, 224, 224), dtype=np.float32)

            # --- STATE PREPARATION (CARTESIAN) ---
            try:
                current_pose = np.array(panda.get_pose()).reshape(4, 4)
                pos = current_pose[:3, 3]
                euler = R.from_matrix(current_pose[:3, :3]).as_euler('xyz')
                
                if gripper.read_once().is_grasped:
                    grip_state = 1.0
                else:
                    grip_state = 0.0
                
                # 8-Dimension State: [X, Y, Z, Roll, Pitch, Yaw, Finger1, Finger2]
                state_8d = np.concatenate([pos, euler, [grip_state, grip_state]]).astype(np.float32)
                state_8d_norm = (state_8d - STATE_MEAN) / STATE_STD

                state_32d = np.zeros(32, dtype=np.float32)
                state_32d[:8] = state_8d_norm

            except Exception as e:
                print(f"[Brain] Error reading panda state: {e}")
                continue

            obs_dict = {
                "observation.state": torch.from_numpy(state_32d).unsqueeze(0).to(device) if "cuda" not in str(device) else torch.from_numpy(state_32d),
                "observation.images.image": torch.from_numpy(model_3rd),
                "observation.images.image2": torch.from_numpy(model_wrist),
                "observation.images.empty_camera_0": torch.from_numpy(dummy_camera),
                "task": INSTRUCTION
            }

            start_time = time.time()
            with torch.inference_mode():
                processed_obs = preprocess(obs_dict)
                action_chunk = policy.select_action(processed_obs)
                processed_action = postprocess(action_chunk)

            interference_time = time.time() - start_time
            action_np = processed_action.cpu().numpy()[0]

            if interference_time > 0.02:
                chunk_id += 1
                step_in_chunk = 0
            else:
                step_in_chunk += 1

            tracked_action = (chunk_id, step_in_chunk, action_np)

            if action_queue.full():
                action_queue.get() 

            action_queue.put(tracked_action)

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
    gripper_is_closed = False

    current_chunk_id = -1
    frame_offset_pos = np.zeros(3)
    frame_offset_rot = R.identity()

    while is_running:
        try:
            chunk_id, step_in_chunk, current_action = action_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # 1. Un-normalize output back into Absolute Cartesian Coordinates
        current_action = (current_action * ACTION_STD) + ACTION_MEAN
        target_x, target_y, target_z, target_roll, target_pitch, target_yaw, target_grip = current_action[:7]

        virtual_pos = np.array([target_x, target_y, target_z])
        virtual_rot = R.from_euler('xyz', [target_roll, target_pitch, target_yaw], degrees=False)

        # 2. ANCHOR THE TRAJECTORY (Apply the Simulator-to-Real-World Offset)
        if chunk_id != current_chunk_id:
            print(f"\n[Muscle] >>> ANCHORING NEW TRAJECTORY CHUNK: #{chunk_id} <<<")
            current_chunk_id = chunk_id
            
            try:
                current_pose = np.array(panda.get_pose()).reshape(4, 4)
                physical_pos = current_pose[:3, 3]
                physical_rot = R.from_matrix(current_pose[:3, :3])
                
                frame_offset_pos = physical_pos - virtual_pos
                frame_offset_rot = physical_rot * virtual_rot.inv()
            except Exception as e:
                print(f"[Muscle] Offset Error: {e}")
                continue

        # Map virtual absolute curve into real-world physical space
        target_pos = virtual_pos + frame_offset_pos
        target_rot = frame_offset_rot * virtual_rot

        # 3. SAFETY CLAMPS
        try:
            current_pose = np.array(panda.get_pose()).reshape(4, 4)
            current_pos = current_pose[:3, 3]
        except Exception:
            continue
            
        step_delta_pos = target_pos - current_pos
        step_delta_pos = np.clip(step_delta_pos, -0.05, 0.05) 
        safe_target_pos = current_pos + step_delta_pos

        # Kill switch for unsafe workspace boundaries
        if (safe_target_pos[0] > 0.7 or safe_target_pos[0] < 0 
            or safe_target_pos[1] > 0.3 or safe_target_pos[1] < -0.3 
            or safe_target_pos[2] > 0.65 or safe_target_pos[2] < 0.05): 
            print("[Warning] Movement rejected: Attempted to move outside safe box limits.")
            continue

        # if step_in_chunk % 5 == 0:
        print(f"[Muscle] Chunk {chunk_id} Step {step_in_chunk} | Target X: {safe_target_pos[0]:.3f}, Y: {safe_target_pos[1]:.3f}, Z: {safe_target_pos[2]:.3f} | Grip: {target_grip:.2f}")

        # 4. EXECUTE PHYSICAL MOVEMENT
        target_pose_mat = np.eye(4)
        target_pose_mat[:3, :3] = target_rot.as_matrix()
        target_pose_mat[:3, 3]  = safe_target_pos
        
        try:
            # panda.move_to_pose(target_pose_mat, speed_factor=0.05)
            
            # Gripper Logic (0.0 is open, 1.0 is closed)
            should_close = target_grip > 0.5 
            
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

    print("[*] Booting Pi0 Asynchronous Framework...")
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