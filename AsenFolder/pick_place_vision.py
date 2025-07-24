#!/usr/bin/env python3
"""
pick_place_vision.py – Pick and Place with Vision using robosuite

This script demonstrates:
• Using camera observations to locate objects
• State machine for pick and place behavior
• Visual feedback processing
• Coordinated arm and gripper control
"""

import time
import numpy as np
import cv2
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ────────────────────────── Configuration ──────────────────────────
CONTROL_FREQ = 20              # Hz
HORIZON = 10000                # steps (no timeout)
CUBE_SIZE = 0.025              # cube size in meters

# Task states
class TaskState:
    APPROACH = "approach"
    DESCEND = "descend" 
    GRASP = "grasp"
    LIFT = "lift"
    MOVE_TO_TARGET = "move_to_target"
    RELEASE = "release"
    RETREAT = "retreat"
    DONE = "done"

# ────────────────────────── Environment Setup ──────────────────────────
def make_env():
    """Create robosuite environment with camera observations"""
    arm_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
    ctrl_cfg = refactor_composite_controller_config(
        arm_cfg, robot_type="Panda", arms=["right"]
    )
    
    raw_env = suite.make(
        "Lift",                    # Task with a cube to pick up
        robots="Panda",
        controller_configs=ctrl_cfg,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        has_renderer=True,
        render_camera="frontview",
        use_camera_obs=True,       # Enable camera observations
        camera_names=["frontview", "birdview"],  # Multiple cameras
        camera_heights=256,
        camera_widths=256,
        reward_shaping=True,
        # Customize cube placement
        placement_initializer=None,  # We'll set this manually after env creation
    )
    
    # Set cube to be closer and more visible
    wrapped_env = GymWrapper(raw_env)
    
    # Access the environment to modify cube placement
    if hasattr(wrapped_env.env, 'placement_initializer'):
        # Set cube position to be more visible and reachable
        wrapped_env.env.cube_pos = [0.1, 0.0, 0.8]  # Closer to robot, on table
    
    return wrapped_env

# ────────────────────────── Vision Processing ──────────────────────────
def find_cube_position(rgb_image, depth_image=None):
    """
    Simple cube detection using color thresholding
    Returns (x, y) pixel coordinates of cube center, or None if not found
    """
    # Convert RGB to HSV for better color detection
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Red cube detection - more lenient ranges
    # HSV ranges for red objects (broader range)
    lower_red1 = np.array([0, 50, 50])      # Lower saturation/value thresholds
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 50, 50])    # Broader red range
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (assumed to be the cube)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 50:  # Lower area threshold
            # Calculate centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), area
    
    return None, 0

def pixel_to_world_position(pixel_x, pixel_y, env, camera_name="frontview"):
    """
    Convert pixel coordinates to approximate world coordinates
    This is a simplified conversion - in practice you'd use proper camera calibration
    """
    # Get cube position from environment (for reference/debugging)
    cube_pos = env.env._get_observations()["cube_pos"]
    
    # Simple mapping (this is approximate - robosuite has better methods)
    # Map pixel coordinates to workspace coordinates
    img_width, img_height = 256, 256
    
    # Approximate workspace bounds (adjust based on your setup)
    x_min, x_max = -0.3, 0.3
    y_min, y_max = -0.2, 0.4
    
    # Convert pixel to world coordinates
    world_x = x_min + (pixel_x / img_width) * (x_max - x_min)
    world_y = y_max - (pixel_y / img_height) * (y_max - y_min)  # Flip Y
    
    return world_x, world_y

# ────────────────────────── Joint Space Poses ──────────────────────────
# Pre-defined joint configurations for different phases
HOME = np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.8, 0.0])
OBSERVE = np.array([0.0, -0.1, 0.0, -1.9, 0.0, 1.7, 0.0])  # Better viewing angle - closer to table

def joints_for_position(x, y, z, env):
    """
    Convert Cartesian position to joint angles
    This is a simplified version - robosuite has built-in IK solvers
    """
    # More conservative poses that keep the robot lower and safer
    if abs(x) < 0.2 and y > -0.1:  # Object roughly in front (more lenient)
        if z > 0.85:  # High position
            return np.array([0.1, 0.0, 0.0, -1.5, 0.0, 1.5, 0.2])
        else:  # Low position (for grasping) - closer to table
            return np.array([0.1, 0.2, 0.0, -1.6, 0.0, 1.8, 0.2])
    else:  # Default safe pose
        return OBSERVE

def joints_for_target_position():
    """Return joint configuration for target drop location"""
    return np.array([-0.3, 0.2, 0.3, -1.6, 0.0, 1.8, -0.2])  # More conservative drop position

# ────────────────────────── Main Control Loop ──────────────────────────
def main():
    env = make_env()
    robot = env.env.robots[0]
    arm = robot.arms[0]
    
    # Reset environment
    obs = env.reset()
    
    # Initialize state machine
    state = TaskState.APPROACH
    state_timer = 0.0
    last_time = time.time()
    
    # Control parameters
    gripper_open = 0.04
    gripper_closed = 0.0
    current_gripper = gripper_open
    
    # Target positions
    cube_pixel_pos = None
    cube_world_pos = None
    
    print("Starting Pick and Place with Vision...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Update timing
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            state_timer += dt
            
            # Get current observations
            obs, _, terminated, truncated, _ = env.step(
                robot.create_action_vector({arm: HOME, f"{arm}_gripper": np.array([current_gripper])})
            )
            
            # Process camera images and get cube position
            cube_pixel_pos = None
            cube_area = 0
            
            # Try vision-based detection first
            if "frontview_image" in obs:
                rgb_image = obs["frontview_image"]
                cube_pixel_pos, cube_area = find_cube_position(rgb_image)
                
                if cube_pixel_pos:
                    cube_world_x, cube_world_y = pixel_to_world_position(
                        cube_pixel_pos[0], cube_pixel_pos[1], env
                    )
                    cube_world_pos = (cube_world_x, cube_world_y, 0.82)  # Approximate table height
                    print(f"Vision: Cube detected at pixel {cube_pixel_pos}, world pos approx {cube_world_pos}")
            
            # Fallback: Use environment's true cube position
            if not cube_pixel_pos:
                try:
                    # Get the actual cube position from the environment
                    env_obs = env.env._get_observations()
                    if "cube_pos" in env_obs:
                        true_cube_pos = env_obs["cube_pos"]
                        cube_world_pos = (true_cube_pos[0], true_cube_pos[1], true_cube_pos[2])
                        print(f"Fallback: Using true cube position {cube_world_pos}")
                        cube_pixel_pos = (128, 128)  # Dummy pixel position
                except:
                    print("Could not get cube position from environment")
            
            # State machine
            if state == TaskState.APPROACH:
                print("State: APPROACH - Moving to observation position")
                target_joints = OBSERVE
                current_gripper = gripper_open
                
                if state_timer > 3.0:  # Give time to reach position
                    if cube_pixel_pos:
                        state = TaskState.DESCEND
                        state_timer = 0.0
                        print("Cube found! Moving to DESCEND")
                    else:
                        print("Cube not detected, continuing search...")
            
            elif state == TaskState.DESCEND:
                print("State: DESCEND - Moving down towards cube")
                if cube_world_pos:
                    target_joints = joints_for_position(cube_world_pos[0], cube_world_pos[1], 0.82, env)
                else:
                    target_joints = OBSERVE
                current_gripper = gripper_open
                
                if state_timer > 2.5:
                    state = TaskState.GRASP
                    state_timer = 0.0
            
            elif state == TaskState.GRASP:
                print("State: GRASP - Closing gripper")
                target_joints = joints_for_position(cube_world_pos[0], cube_world_pos[1], 0.82, env) if cube_world_pos else OBSERVE
                current_gripper = gripper_closed
                
                if state_timer > 1.5:
                    state = TaskState.LIFT
                    state_timer = 0.0
            
            elif state == TaskState.LIFT:
                print("State: LIFT - Lifting cube")
                target_joints = joints_for_position(cube_world_pos[0], cube_world_pos[1], 0.95, env) if cube_world_pos else OBSERVE
                current_gripper = gripper_closed
                
                if state_timer > 2.0:
                    state = TaskState.MOVE_TO_TARGET
                    state_timer = 0.0
            
            elif state == TaskState.MOVE_TO_TARGET:
                print("State: MOVE_TO_TARGET - Moving to drop location")
                target_joints = joints_for_target_position()
                current_gripper = gripper_closed
                
                if state_timer > 3.0:
                    state = TaskState.RELEASE
                    state_timer = 0.0
            
            elif state == TaskState.RELEASE:
                print("State: RELEASE - Opening gripper")
                target_joints = joints_for_target_position()
                current_gripper = gripper_open
                
                if state_timer > 1.5:
                    state = TaskState.RETREAT
                    state_timer = 0.0
            
            elif state == TaskState.RETREAT:
                print("State: RETREAT - Moving to home")
                target_joints = HOME
                current_gripper = gripper_open
                
                if state_timer > 2.5:
                    state = TaskState.DONE
                    state_timer = 0.0
            
            elif state == TaskState.DONE:
                print("Task completed! Holding at HOME position")
                target_joints = HOME
                current_gripper = gripper_open
                
                if state_timer > 5.0:  # Reset after 5 seconds
                    state = TaskState.APPROACH
                    state_timer = 0.0
                    print("Restarting task...")
            
            # Send action to robot
            action = robot.create_action_vector({
                arm: target_joints,
                f"{arm}_gripper": np.array([current_gripper])
            })
            env.step(action)
            env.render()
            
            # Check for termination
            if terminated or truncated:
                print("Episode ended, resetting...")
                obs = env.reset()
                state = TaskState.APPROACH
                state_timer = 0.0
    
    except KeyboardInterrupt:
        print("\nTask interrupted by user")
    finally:
        env.close()
        print("Pick and Place with Vision demo finished.")

if __name__ == "__main__":
    main()
