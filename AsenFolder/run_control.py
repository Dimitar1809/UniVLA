#!/usr/bin/env python3
"""
run_control.py – drive a Franka Panda in robosuite ≥ 1.5 with absolute
7-D joint-angle targets (the format UniVLA outputs).

Changes vs. previous version
----------------------------
• `horizon=1000` so the episode no longer truncates after 5 s.
• Minor logging of `terminated` / `truncated` at the end for clarity.
"""

import time
import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ────────────────────────── joint-space way-points ──────────────────────────
HOME      = np.array([0.00, -0.50, 0.00, -2.20, 0.00, 1.80, 0.00])
PICK_PREP = np.array([0.20,  0.30, 0.00, -1.50, 0.00, 1.70, 0.50])
PICK      = np.array([0.20,  0.40, 0.00, -1.80, 0.00, 2.00, 0.50])
LIFT      = np.array([0.25, -0.10, 0.00, -1.00, 0.00, 1.50, 0.30])
DROP_PREP = np.array([-0.50, 0.25, 0.50, -1.50, 0.00, 1.50, -0.30])
DROP      = np.array([-0.50, 0.35, 0.50, -1.80, 0.00, 1.80, -0.30])

WAYPOINTS      = [HOME, PICK_PREP, PICK, LIFT, DROP_PREP, DROP, HOME]
GRIPPER_CMDS   = [0.04, 0.04, 0.00, 0.00, 0.00, 0.04, 0.04]   # open⇢close
SEGMENT_TIMES  = [2.5, 2.0, 1.0, 2.0, 2.5, 1.0]               # seconds

CONTROL_FREQ   = 10                                           # Hz
HORIZON        = 10000                                         # steps

# ─────────────────────────── environment helper ────────────────────────────
def make_env():
    # arm-only JOINT_POSITION PD  →  composite format
    arm_cfg  = load_part_controller_config(default_controller="JOINT_POSITION")
    ctrl_cfg = refactor_composite_controller_config(
        arm_cfg, robot_type="Panda", arms=["right"]
    )

    raw_env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=ctrl_cfg,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        has_renderer=True,
        render_camera="frontview",
        use_camera_obs=False,
        reward_shaping=True,
    )
    return GymWrapper(raw_env)

# ───────────────────────────── rollout loop ────────────────────────────────
def main():
    env          = make_env()
    sim_env      = env.env
    robot        = sim_env.robots[0]
    arm          = robot.arms[0]
    observation  = env.reset()

    seg_idx, seg_time = 0, 0.0
    last_wall         = time.time()
    terminated = truncated = False

    while seg_idx < len(SEGMENT_TIMES) and not (terminated or truncated):
        # real-time pacing (optional)
        now, dt = time.time(), time.time() - last_wall
        last_wall, seg_time = now, seg_time + dt

        # linear interpolation between way-points
        alpha  = min(1.0, seg_time / SEGMENT_TIMES[seg_idx])
        q_des  = (1 - alpha) * WAYPOINTS[seg_idx] + alpha * WAYPOINTS[seg_idx + 1]

        action = robot.create_action_vector({
            arm: q_des,
            f"{arm}_gripper": np.array([GRIPPER_CMDS[seg_idx]]),
        })

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if seg_time >= SEGMENT_TIMES[seg_idx]:
            seg_idx  += 1
            seg_time  = 0.0

    # optional 2-s hold at HOME if not already ended
    if not (terminated or truncated):
        for _ in range(int(2 * CONTROL_FREQ)):
            env.step(robot.create_action_vector({arm: HOME, f"{arm}_gripper": np.array([0.04])}))
            env.render()

    env.close()
    print(f"Simulation finished. terminated={terminated}, truncated={truncated}")

if __name__ == "__main__":
    main()
