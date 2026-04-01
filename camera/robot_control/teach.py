import logging
import time
import panda_py
from panda_py import controllers

# 1. Setup & Credentials
hostname = '172.16.0.2'
username = 'Dentec'
password = 'Frankenstein'

logging.basicConfig(level=logging.INFO)

# 2. Unlock and Initialize
# Note: Unlock/Activate FCI is required before the Panda object can take control
desk = panda_py.Desk(hostname, username, password)
desk.unlock()
desk.activate_fci()

panda = panda_py.Panda(hostname)
panda.move_to_start(speed_factor=0.05)

# --- PART 1: Teach 3 Discrete Joint Positions ---
print('--- Teaching Mode: Poses ---')
positions = []
panda.teaching_mode(True) # Arm becomes compliant

for i in range(3):
    input(f'Manually move the arm to Pose {i+1} and press Enter...')
    positions.append(panda.q) # Capture current joint angles

panda.teaching_mode(False) # Re-engage brakes/stiffness

input('Press Enter to replay the 3 poses...')
# move_to_joint_position can accept a list of positions to visit sequentially
panda.move_to_joint_position(positions, speed_factor=0.1)

# # --- PART 2: Teach & Replay a Trajectory ---
# print('\n--- Teaching Mode: Trajectory ---')
# RECORD_SECONDS = 5
# input(f'Press Enter to record {RECORD_SECONDS}s of movement...')

# panda.teaching_mode(True)
# panda.enable_logging(RECORD_SECONDS * 1000) # Buffer size in ms
# time.sleep(RECORD_SECONDS)
# panda.teaching_mode(False)

# # Retrieve the recorded joint positions (q) and velocities (dq)
# log_data = panda.get_log()
# q_recorded = log_data['q']
# dq_recorded = log_data['dq']

# input('Press Enter to replay the recorded trajectory...')
# # Move to the starting point of the trajectory first
# panda.move_to_joint_position(q_recorded[0], speed_factor=0.1)

# # Initialize controller for smooth playback
# i = 0
# ctrl = controllers.JointPosition()
# panda.start_controller(ctrl)

# # Playback loop at 1kHz (1000Hz)
# with panda.create_context(frequency=1000, max_runtime=RECORD_SECONDS) as ctx:
#     while ctx.ok() and i < len(q_recorded):
#         ctrl.set_control(q_recorded[i], dq_recorded[i])
#         i += 1
        
# print("Demonstration complete.")