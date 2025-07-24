#!/usr/bin/env python3
"""
raw_mj_step.py – simplest “down & up” motion driven by mujoco.mj_step()

No Gym, no robosuite env layer – we talk straight to MuJoCo.
"""

import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.empty_arena import EmptyArena
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory

# ----------------------------------------------------------------------
# 1) Build world: empty floor + Panda with a gripper (no objects needed)
# ----------------------------------------------------------------------
world = MujocoWorldBase()
world.merge(EmptyArena())

panda = Panda()
panda.add_gripper(gripper_factory("PandaGripper"))   # optional (keeps same nu)
world.merge(panda)

model = world.get_model(mode="mujoco")
data  = mujoco.MjData(model)

# ----------------------------------------------------------------------
# 2) Configure actuator gains for proper position control
# ----------------------------------------------------------------------
# Set high position gains (kp) and reasonable damping (kv) for arm joints
# These values are similar to what robosuite uses internally
arm_kp = 400.0   # position gain
arm_kv = 40.0    # velocity gain (damping)

for i in range(7):  # First 7 actuators are arm joints
    if i < model.nu:
        model.actuator_gainprm[i, 0] = arm_kp  # kp gain
        model.actuator_biasprm[i, 1] = -arm_kv  # kv gain (damping)

# Set lower gains for gripper actuators
gripper_kp = 200.0
gripper_kv = 20.0
for i in range(7, min(9, model.nu)):  # Gripper actuators
    model.actuator_gainprm[i, 0] = gripper_kp
    model.actuator_biasprm[i, 1] = -gripper_kv

# Add joint damping to help stabilize the robot
for i in range(7):  # First 7 joints are arm joints
    if i < model.nv:
        model.dof_damping[i] = 5.0  # Add damping to arm joints

# ----------------------------------------------------------------------
# 3) Show available actuators so we know their order
# ----------------------------------------------------------------------
print(f"Found {model.nu} actuators:")
for i in range(model.nu):
    print(f"  [{i:2d}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

# The first 7 are Panda joints, the last 2 are the fingers.  All are
# MuJoCo *position* actuators, so data.ctrl[i] is a desired joint angle.

# ----------------------------------------------------------------------
# 4) Define poses (rad) and timing
# ----------------------------------------------------------------------
HOME = np.array([0.00, -0.50, 0.00, -2.20, 0.00, 1.80, 0.00])
DOWN = np.array([0.00,  0.30, 0.00, -2.20, 0.00, 1.80, 0.00])

PHASES = [
    dict(start=HOME, end=DOWN,   dur=3.0),   # go down   3 s
    dict(start=DOWN, end=DOWN,   dur=3.0),   # hold      3 s
    dict(start=DOWN, end=HOME,   dur=3.0),   # go up     3 s
]

# ----------------------------------------------------------------------
# 5) Initialize robot to HOME position before starting
# ----------------------------------------------------------------------
# Set initial joint positions to HOME pose
data.qpos[:7] = HOME  # Set the first 7 joint positions (Panda arm)
data.ctrl[:7] = HOME  # Set initial control targets
data.ctrl[7:] = 0.04  # Keep gripper open

# Forward kinematics to update the robot state
mujoco.mj_forward(model, data)

# Run a few steps to let the robot stabilize at HOME position
print("Stabilizing robot at HOME position...")
for _ in range(100):  # Let the robot settle for ~100 steps
    data.ctrl[:7] = HOME
    data.ctrl[7:] = 0.04
    mujoco.mj_step(model, data)

print("Robot stabilized. Starting motion...")

# ----------------------------------------------------------------------
# 6) Launch non-blocking viewer
# ----------------------------------------------------------------------
viewer = launch_passive(model, data)

# ----------------------------------------------------------------------
# 7) Servo loop
# ----------------------------------------------------------------------
for phase in PHASES:
    t0 = data.time
    while data.time - t0 < phase["dur"]:
        # interpolation factor 0→1 within this phase
        alpha = (data.time - t0) / phase["dur"]
        alpha = np.clip(alpha, 0.0, 1.0)

        # desired joints for the seven arm actuators
        q_des = (1 - alpha) * phase["start"] + alpha * phase["end"]

        data.ctrl[:7] = q_des          # send to position servos
        data.ctrl[7:] = 0.04           # keep gripper open (0.04 m)

        mujoco.mj_step(model, data)    # integrate 1 simulation step
        viewer.sync()                  # update the on-screen viewer

# hold HOME 2 s
hold_until = data.time + 2.0
while data.time < hold_until:
    data.ctrl[:7] = HOME
    data.ctrl[7:] = 0.04
    mujoco.mj_step(model, data)
    viewer.sync()

print("Down-and-up demo finished.")
viewer.close()
