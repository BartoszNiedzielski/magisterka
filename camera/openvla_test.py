import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import time

import logging
import panda_py
from panda_py import libfranka

import numpy as np
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
MODEL_PATH = "openvla/openvla-7b"
INSTRUCTION = "pick up the red ball"


# === ROBOT CONNECTION SETUP ===
# Panda hostname/IP and Desk login information of your robot
hostname = '172.16.0.2'
username = 'Dentec'
password = 'Frankenstein'

# panda-py is chatty, activate information log level
logging.basicConfig(level=logging.INFO)

desk = panda_py.Desk(hostname, username, password)
desk.unlock()
desk.activate_fci()

panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

# Get current joint positions (7 angles)
q = panda.q
print(f"Joint Angles: {q}")

# Get End-Effector Position (3x1 vector)
ee_position = panda.get_position()
print(f"EE Position: {ee_position}")

# Get End-Effector Orientation (Quaternion)
ee_orientation = panda.get_orientation()
print(f"EE Orientation: {ee_orientation}")


# === 1. CAMERA SETUP (RealSense) ===
print("[*] Starting RealSense Pipeline...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# === 2. MODEL SETUP ===
print(f"[*] Loading OpenVLA (`{MODEL_PATH}`) with Flash Attention 2...")

# 4-bit Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    attn_implementation="flash_attention_2",  # ENABLED
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# Move model to CUDA (Usually automatic with bitsandbytes, but good to be explicit)
device = torch.device("cuda")

print(f"\n[*] Ready! Instruction: '{INSTRUCTION}'")
print("[*] Press 'q' to quit.")

try:
    while True:
        # A. Capture Frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        
        # Convert to Numpy (BGR) for OpenCV
        image_np = np.asanyarray(color_frame.get_data())
        
        # Convert to PIL (RGB) for OpenVLA
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        # B. Prepare Inputs
        # Explicit prompt format is safer
        prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
        
        # Note: We use return_tensors="pt" to get tensors immediately
        inputs = processor(prompt, image_pil, return_tensors="pt").to(device, dtype=torch.bfloat16)

        # C. Inference
        start_time = time.time()
        with torch.inference_mode():
            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        dt = time.time() - start_time

        dx, dy, dz, droll, dpitch, dyaw, grip = action
        print(f"Predicted Action: [{dx:.3f}, {dy:.3f}, {dz:.3f}, {droll:.3f}, {dpitch:.3f}, {dyaw:.3f}, {grip:.3f}] (Inference Time: {dt:.3f}s)")
        # dx = dx *-1
        # dy = dy *-1

        # Visualize of predicted action (dx, dy, dz, droll, dpitch, dyaw, grip)
        h, w = image_np.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Scale for visibility (x3000 is typical for small delta actions)
        u = int(dy * 3000)
        v = int(-dz * 3000) 

        # Draw Arrow & Text
        cv2.arrowedLine(image_np, (cx, cy), (cx + u, cy + v), (0, 255, 255), 4)
        
        grip_txt = "GRASP" if grip < 0.5 else "OPEN"
        color = (0, 0, 255) if grip < 0.5 else (0, 255, 0)
        cv2.putText(image_np, f"{grip_txt} ({dt:.3f}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display
        cv2.imshow("OpenVLA + FlashAttn", image_np)

        # print(f"Predicted Action: {action} (Inference Time: {dt:.3f}s)")

        # Move the robot if within safe limits
        # dx, dy, dz, droll, dpitch, dyaw, grip = action
        # dx = dx *-1
        # dy = dy *-1
        
        # Clip deltas to prevent large jumps (safety measure)
        dx = np.clip(dx, -0.05, 0.05)
        dy = np.clip(dy, -0.05, 0.05)
        dz = np.clip(dz, -0.05, 0.05)

        current_pose = np.array(panda.get_pose()).reshape(4, 4)

        # Extract current position (Vector) and rotation (Matrix)
        current_pos = current_pose[:3, 3]
        current_rot_mat = current_pose[:3, :3]
        current_rot = R.from_matrix(current_rot_mat)

        # 2. Calculate Target Translation
        # Apply the delta to the current base-frame position
        target_pos = current_pos + np.array([dx, dy, dz])

        # 3. Calculate Target Rotation
        # OpenVLA outputs Roll, Pitch, Yaw in radians. 
        # 'xyz' denotes extrinsic rotations standard to the Bridge dataset.
        delta_rot = R.from_euler('xyz', [droll, dpitch, dyaw], degrees=False)

        # Pre-multiply to apply the rotation delta in the global/base frame
        target_rot = delta_rot * current_rot 

        # 4. Construct the new 4x4 Target Pose Matrix
        target_pose = np.eye(4)
        target_pose[:3, :3] = target_rot.as_matrix()
        target_pose[:3, 3]  = target_pos

        # 5. Execute the move
        # Flatten the 4x4 matrix back to a 16-element list for panda-py
        print(f"Moving to delta: [{dx:.3f}, {dy:.3f}, {dz:.3f}]")

        if (target_pos[0] > 0.7 or target_pos[0] < 0 # X-axis limits
            or target_pos[1] > 0.3 or target_pos[1] < -0.3 # Y-axis limits
            or target_pos[2] > 0.65 or target_pos[2] < 0.1): # Z-axis limits    
            print("Warning: Attempted to move beyond safe box limits.")
            continue  # Skip moving if out of bounds

        print(f"Moving to new position: {target_pos}")
        panda.move_to_pose(target_pose, speed_factor=0.05)
        
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    desk.lock()
    desk.release_control()