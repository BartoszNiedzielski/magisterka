import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import threading
import logging
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import panda_py
from panda_py import libfranka

import numpy as np
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'

MODEL_PATH = "openvla/openvla-7b"
INSTRUCTION = "pick up the green cube"
ACTION_SCALE = 1

# panda-py is chatty, activate information log level
logging.basicConfig(level=logging.INFO)

# === SHARED VARIABLES (The Bridge between Brain and Muscle) ===
# We use a Lock to prevent both threads from writing/reading at the exact same millisecond
state_lock = threading.Lock()
latest_dx = 0.0
latest_dy = 0.0
latest_dz = 0.0
latest_dpitch = 0.0
latest_droll = 0.0
latest_dyaw = 0.0
latest_grip = 1.0
is_running = True  # Used to shut down both threads cleanly

# ==========================================
# THREAD 1: THE BRAIN (Vision & AI)
# ==========================================
def vision_loop():
    global latest_dx, latest_dy, latest_dz, latest_grip, is_running
    
    print("[Brain] Starting Camera and AI...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # === full model without quantization (uncomment if you have >24GB GPU RAM and want the best performance) ===
    # device = torch.device("cuda")
    # processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # # Loading the full, original 7B model in 16-bit precision
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH, 
    #     attn_implementation="flash_attention_2", 
    #     torch_dtype=torch.bfloat16,  # 16-bit precision (Native)
    #     low_cpu_mem_usage=True, 
    #     trust_remote_code=True
    # ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        quantization_config=bnb_config, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True)
    device = torch.device("cuda")

    print("[Brain] AI Online. Listening for visual updates...")
    
    try:
        while is_running:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            image_np = np.asanyarray(color_frame.get_data())
            image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

            prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
            inputs = processor(prompt, image_pil, return_tensors="pt").to(device, dtype=torch.bfloat16)

            start_time = time.time()
            with torch.inference_mode():
                action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            dx_ai, dy_ai, dz_ai, droll, dpitch, dyaw, grip = action
            print(f"[Brain] Predicted Action: dx={dx_ai:.3f}, dy={dy_ai:.3f}, dz={dz_ai:.3f}, droll={droll:.3f}, dpitch={dpitch:.3f}, dyaw={dyaw:.3f}, grip={grip:.3f} (Latency: {time.time() - start_time:.3f}s)")

            # Update the shared variables safely
            with state_lock:
                latest_dx = dx_ai * ACTION_SCALE
                latest_dy = dy_ai * ACTION_SCALE
                latest_dz = dz_ai * ACTION_SCALE
                latest_droll = droll * ACTION_SCALE
                latest_dpitch = dpitch * ACTION_SCALE
                latest_dyaw = dyaw * ACTION_SCALE
                latest_grip = grip

            # Visualization (Optional, but good for debugging)
            h, w = image_np.shape[:2]
            cx, cy = w // 2, h // 2
            u = int(dy_ai * 3000)
            v = int(-dz_ai * 3000)
            cv2.arrowedLine(image_np, (cx, cy), (cx + u, cy + v), (0, 255, 255), 4)
            cv2.putText(image_np, f"Brain Latency: {time.time() - start_time:.3f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("OpenVLA Vision Thread", image_np)
            
            if cv2.waitKey(1) == ord('q'):
                is_running = False
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Brain] Shutting down.")


# ==========================================
# THREAD 2: THE MUSCLE (Robot Control)
# ==========================================
def control_loop():
    global is_running
    
    print("[Muscle] Authenticating with Franka Desk...")
    try:
        # 1. Unlock and setup FCI
        desk = panda_py.Desk(ROBOT_IP, ROBOT_USER, ROBOT_PASS)
        desk.unlock()
        desk.activate_fci()
        print("[Muscle] Brakes unlocked and FCI activated.")

        # 2. Connect to the robot hardware
        panda = panda_py.Panda(ROBOT_IP)
        gripper = libfranka.Gripper(ROBOT_IP)
        
        print("[Muscle] Homing robot...")
        panda.move_to_start(speed_factor=0.05)

        pose = panda.get_pose()
        pose[2,3] -= 0.2
        q = panda_py.ik(pose)
        panda.move_to_joint_position(q, speed_factor=0.05)
        print(panda.get_position())
        time.sleep(1)
        
        print("[Muscle] Robot Online. Ready to move.")

    except Exception as e:
        print(f"[Muscle] Failed to connect to Robot: {e}")
        is_running = False  # Shut down the whole script if connection fails
        return

    while is_running:
        # Read AI command
        with state_lock:
            target_dx = np.clip(latest_dx, -0.05, 0.05)
            target_dy = np.clip(latest_dy, -0.05, 0.05)
            target_dz = np.clip(latest_dz, -0.05, 0.05)
            target_droll = np.clip(latest_droll, -0.1, 0.1)
            target_dpitch = np.clip(latest_dpitch, -0.1, 0.1)
            target_dyaw = np.clip(latest_dyaw, -0.1, 0.1)
            target_grip = latest_grip

        # Get current physical position
        current_pose = np.array(panda.get_pose()).reshape(4, 4)
        current_pos = current_pose[:3, 3]
        current_rot_mat = current_pose[:3, :3]
        current_rot = R.from_matrix(current_rot_mat)

        delta_rot = R.from_euler('xyz', [target_droll, target_dpitch, target_dyaw], degrees=False)

        # Pre-multiply to apply the rotation delta in the global/base frame
        target_rot = delta_rot * current_rot

        # Calculate micro-target
        target_pos = current_pos + np.array([target_dx, target_dy, target_dz])

        if (target_pos[0] > 0.7 or target_pos[0] < 0 # X-axis limits
            or target_pos[1] > 0.3 or target_pos[1] < -0.3 # Y-axis limits
            or target_pos[2] > 0.65 or target_pos[2] < 0.05): # Z-axis limits    
            print("Warning: Attempted to move beyond safe box limits.")
            continue  # Skip moving if out of bounds
        
        target_pose = np.eye(4)
        target_pose[:3, :3] = target_rot.as_matrix()
        target_pose[:3, 3]  = target_pos

        # Execute smooth micro-movement
        panda.move_to_pose(target_pose, speed_factor=0.05)
        
        # Gripper logic (libfranka specific)
        try:
            if target_grip < 0.5:
                print("[Muscle] Closing gripper.")
                # Close gripper (libfranka syntax)
                gripper.grasp(width=0.0, speed=0.1, force=40)
            else:
                # Open gripper (libfranka syntax)
                gripper.move(width=0.08, speed=0.1)
        except Exception as e:
            # Gripper might throw an exception if it's already fully closed/open
            pass

        # Small sleep prevents the control loop from hogging the CPU
        time.sleep(0.1) 
    
    print("[Muscle] Shutting down.")
    desk.lock()
    desk.release_control()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[*] Booting Asynchronous Robotics System...")
    
    # Create the threads
    brain_thread = threading.Thread(target=vision_loop)
    muscle_thread = threading.Thread(target=control_loop)
    
    # Start them simultaneously
    brain_thread.start()
    muscle_thread.start()
    
    # Keep the main script alive until threads finish
    brain_thread.join()
    muscle_thread.join()
    
    print("[*] System safely powered down.")