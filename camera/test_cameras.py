import time
import cv2
import numpy as np
import threading
import logging
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
ROBOT_IP = '172.16.0.2'
ROBOT_USER = 'Dentec'
ROBOT_PASS = 'Frankenstein'
INSTRUCTION = "pick up the green cube"

# === MULTI-CAMERA SETUP ===
# Change these indices if OpenCV grabs the wrong USB feeds
CAM_INDEX_3RD_PERSON = 0  
CAM_INDEX_WRIST = 2       

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
    print("2")
    cap_wrist = cv2.VideoCapture(CAM_INDEX_WRIST)
    print("1")
    
    try:
        while is_running:
            # Grab frames
            ret1, frame_3rd = cap_3rd.read()
            ret2, frame_wrist = cap_wrist.read()
            if not ret1 or not ret2: continue
            # openpi expects standard RGB Numpy arrays (not BGR from OpenCV, and not PyTorch Tensors!)
            img_3rd_rgb = cv2.cvtColor(frame_3rd, cv2.COLOR_BGR2RGB)
            img_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
            combined_view = np.hstack((frame_3rd, frame_wrist))
            # cv2.imshow("Robot Vision: 3rd Person (Left) | Wrist (Right)", combined_view)
            
            # if cv2.waitKey(1) == ord('q'):
            #     is_running = False
            #     break
                
    finally:
        cap_3rd.release()
        cap_wrist.release()
        cv2.destroyAllWindows()
        print("[Brain] Shutting down.")

if __name__ == "__main__":
    print("[*] Booting Pi0.5 Asynchronous Framework...")
    brain_thread = threading.Thread(target=vision_loop)
    brain_thread.start()
    brain_thread.join()
    print("[*] System safely powered down.")