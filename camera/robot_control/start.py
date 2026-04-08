# Panda hostname/IP and Desk login information of your robot
hostname = '172.16.0.2'
username = 'Dentec'
password = 'Frankenstein'

# panda-py is chatty, activate information log level
import logging
logging.basicConfig(level=logging.INFO)

import panda_py

desk = panda_py.Desk(hostname, username, password)
desk.unlock()
desk.activate_fci()

from panda_py import libfranka

panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

panda.move_to_start(speed_factor=0.05)
pose = panda.get_pose()
pose[2,3] -= 0.1
q = panda_py.ik(pose)
panda.move_to_joint_position(q, speed_factor=0.05)
# print(panda.get_position())