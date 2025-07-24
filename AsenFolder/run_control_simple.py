#!/usr/bin/env python3
"""
run_simple.py ─ a *very* simple trajectory for a Franka Panda arm in
robosuite ≥ 1.5:  HOME  →  DOWN  →  HOME.

• Uses the built-in JOINT_POSITION PD (absolute joint targets).
• Episode horizon set long enough that time-outs never trigger.
"""

import time
import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ─────────── two joint-space poses (rad) ───────────
HOME = np.array([0.00, -0.50, 0.00, -2.20, 0.00, 1.80, 0.00])
DOWN = np.array([0.00, +0.30, 0.00, -2.20, 0.00, 1.80, 0.00])   # joints 2↑ to dip wrist

POSES         = [HOME, DOWN, HOME]
SEG_DURATIONS = [3.0, 5.0]      # 3 s down, 1 s back up
CONTROL_FREQ  = 20              # Hz
HORIZON       = 10000            # plenty of steps

# ─────────── environment factory ───────────
def make_env():
    arm_cfg  = load_part_controller_config(default_controller="JOINT_POSITION")
    ctrl_cfg = refactor_composite_controller_config(
        arm_cfg, robot_type="Panda", arms=["right"]
    )
    raw_env = suite.make(
        "Lift",                 # any task works—we ignore the cube
        robots="Panda",
        controller_configs=ctrl_cfg,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        has_renderer=True,
        render_camera="frontview",
        use_camera_obs=False,
        reward_shaping=False,
    )
    return GymWrapper(raw_env)

# ─────────── rollout ───────────
def main():
    env   = make_env()
    robot = env.env.robots[0]
    arm   = robot.arms[0]
    env.reset()

    seg_i, t_seg, last = 0, 0.0, time.time()
    terminated = truncated = False

    while seg_i < len(SEG_DURATIONS) and not (terminated or truncated):
        # wall-clock pacing
        now, dt = time.time(), time.time() - last
        last, t_seg = now, t_seg + dt

        # linear interp between current and next pose
        alpha = min(1.0, t_seg / SEG_DURATIONS[seg_i])
        q_des = (1 - alpha) * POSES[seg_i] + alpha * POSES[seg_i + 1]

        action = robot.create_action_vector({arm: q_des})
        _, _, terminated, truncated, _ = env.step(action)
        env.render()

        if t_seg >= SEG_DURATIONS[seg_i]:
            seg_i += 1
            t_seg  = 0.0

    env.close()
    print("Simple down-and-up demo finished.")

if __name__ == "__main__":
    main()
