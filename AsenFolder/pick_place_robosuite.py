#!/usr/bin/env python3
"""
pick_place_robosuite.py – Pick and Place with a better predefined policy

This version fixes the observation parsing and uses a more direct approach
with proper gripper control.
"""

import numpy as np
import robosuite as suite

class BetterPickPlacePolicy:
    """
    A better predefined policy that properly reads observations
    Note: This environment only has 7-DOF arm control, gripper is automatic
    """
    def __init__(self):
        self.state = "observe"
        self.state_steps = 0
        
    def get_action(self, obs):
        """
        Get action based on current observation and internal state
        """
        # Get end-effector position directly from observations!
        ee_pos = obs.get("robot0_eef_pos", np.array([0.0, 0.0, 1.0]))
        
        # Get object positions - there are multiple objects in the scene
        milk_pos = obs.get("Milk_pos", np.array([0.0, -0.2, 0.88]))
        bread_pos = obs.get("Bread_pos", np.array([0.2, -0.3, 0.85]))
        cereal_pos = obs.get("Cereal_pos", np.array([0.1, -0.1, 0.9]))
        can_pos = obs.get("Can_pos", np.array([0.0, -0.4, 0.86]))
        
        # Pick the closest object as our target
        objects = {"Milk": milk_pos, "Bread": bread_pos, "Cereal": cereal_pos, "Can": can_pos}
        closest_obj = "Milk"  # Default to milk
        min_dist = float('inf')
        for name, pos in objects.items():
            dist = np.linalg.norm(ee_pos - pos)
            if dist < min_dist:
                min_dist = dist
                closest_obj = name
                obj_pos = pos
        
        target_pos = np.array([0.0, 0.2, 0.88])  # Target location on table
        
        # Print debug info every 50 steps
        if self.state_steps % 50 == 0:
            print(f"Step {self.state_steps}: State={self.state} Target={closest_obj}")
            print(f"  EE pos: {ee_pos}")
            print(f"  Obj pos: {obj_pos}")
            print(f"  Distance to obj: {np.linalg.norm(ee_pos - obj_pos):.3f}")
        
        # Simple state machine - only 7 DOF arm control
        action = np.zeros(7)
        
        if self.state == "observe":
            # Hold still for a moment
            if self.state_steps > 10:
                self.state = "approach"
                self.state_steps = 0
                print(f"OBSERVE -> APPROACH (targeting {closest_obj})")
        
        elif self.state == "approach":
            # Move towards object
            approach_pos = obj_pos + np.array([0, 0, 0.1])  # 10cm above object
            error = approach_pos - ee_pos
            
            # Proportional control for each joint (simplified inverse kinematics)
            action[0] = np.clip(error[1] * 2.0, -0.5, 0.5)   # Base joint for Y movement
            action[1] = np.clip(error[2] * 1.5, -0.5, 0.5)   # Shoulder for Z movement
            action[2] = np.clip(error[0] * 1.5, -0.5, 0.5)   # Elbow for X movement
            action[3] = np.clip(-error[2] * 0.5, -0.3, 0.3)  # Wrist 1
            action[4] = np.clip(error[0] * 0.3, -0.3, 0.3)   # Wrist 2
            action[5] = np.clip(error[1] * 0.3, -0.3, 0.3)   # Wrist 3
            action[6] = 0.0  # End effector rotation
            
            if self.state_steps > 100 or np.linalg.norm(error) < 0.08:
                self.state = "descend"
                self.state_steps = 0
                print("APPROACH -> DESCEND")
        
        elif self.state == "descend":
            # Move down to object
            grasp_pos = obj_pos + np.array([0, 0, 0.02])  # Just above object
            error = grasp_pos - ee_pos
            
            # Gentle movement down
            action[1] = np.clip(error[2] * 1.0, -0.2, 0.2)   # Mainly Z movement
            action[0] = np.clip(error[1] * 0.5, -0.2, 0.2)   # Small adjustments
            action[2] = np.clip(error[0] * 0.5, -0.2, 0.2)
            
            if self.state_steps > 50 or (error[2] < 0.05 and np.linalg.norm(error[:2]) < 0.05):
                self.state = "grasp"
                self.state_steps = 0
                print("DESCEND -> GRASP")
        
        elif self.state == "grasp":
            # Hold position briefly (gripper should close automatically)
            action *= 0.1  # Small holding actions
            
            if self.state_steps > 30:
                self.state = "lift"
                self.state_steps = 0
                print("GRASP -> LIFT")
        
        elif self.state == "lift":
            # Move up
            lift_pos = obj_pos + np.array([0, 0, 0.2])  # Lift 20cm
            error = lift_pos - ee_pos
            
            action[1] = np.clip(error[2] * 1.0, -0.3, 0.3)   # Upward movement
            action[0] = np.clip(error[1] * 0.2, -0.1, 0.1)   # Small stabilization
            action[2] = np.clip(error[0] * 0.2, -0.1, 0.1)
            
            if self.state_steps > 60:
                self.state = "place"
                self.state_steps = 0
                print("LIFT -> PLACE")
        
        elif self.state == "place":
            # Move to target location
            place_pos = target_pos + np.array([0, 0, 0.1])  # Above target
            error = place_pos - ee_pos
            
            action[0] = np.clip(error[1] * 1.0, -0.3, 0.3)   # Move to target Y
            action[2] = np.clip(error[0] * 1.0, -0.3, 0.3)   # Move to target X
            action[1] = np.clip(error[2] * 0.5, -0.2, 0.2)   # Z adjustment
            
            if self.state_steps > 80 or np.linalg.norm(error) < 0.1:
                self.state = "drop"
                self.state_steps = 0
                print("PLACE -> DROP")
        
        elif self.state == "drop":
            # Lower the object
            drop_pos = target_pos + np.array([0, 0, 0.02])
            error = drop_pos - ee_pos
            
            action[1] = np.clip(error[2] * 0.8, -0.2, 0.2)   # Gentle lowering
            
            if self.state_steps > 40:
                self.state = "done"
                self.state_steps = 0
                print("DROP -> DONE")
        
        elif self.state == "done":
            # Task complete, minimal movement
            action *= 0
        
        self.state_steps += 1
        return action
    
    def reset(self):
        """Reset policy state for new episode"""
        self.state = "observe"
        self.state_steps = 0

def main():
    """
    Demo of the PickPlace environment with a better predefined policy
    """
    print("Creating PickPlace environment...")
    
    # Create environment following robosuite docs
    env = suite.make(
        env_name="PickPlace",           # The Pick-and-Place task
        robots="Panda",                 # Use Panda robot
        has_renderer=True,              # Enable on-screen rendering
        render_camera="frontview",      # Camera view for rendering
        has_offscreen_renderer=False,   # No off-screen rendering needed
        control_freq=20,                # 20 Hz control
        horizon=500,                    # Episode length
        use_object_obs=True,            # Use object state observations
        use_camera_obs=False,           # No camera observations for now
        reward_shaping=True,            # Use shaped rewards for learning
    )
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_spec}")
    
    # Create policy
    policy = BetterPickPlacePolicy()
    
    # Reset environment
    obs = env.reset()
    print(f"Observation keys: {list(obs.keys())}")
    
    # Show some observation details
    if "object-state" in obs:
        print(f"Object state shape: {obs['object-state'].shape}")
        print(f"Object state sample: {obs['object-state'][:10]}")  # First 10 values
    if "robot0_proprio-state" in obs:
        print(f"Robot proprioception shape: {obs['robot0_proprio-state'].shape}")
        print(f"Robot state sample: {obs['robot0_proprio-state'][:10]}")  # First 10 values
    
    # Debug: Print action spec details
    low, high = env.action_spec
    print(f"Action space low: {low}")
    print(f"Action space high: {high}")
    print(f"Action space size: {len(low)}")
    
    print("\nStarting better predefined policy demo...")
    print("The robot will follow a pick-and-place policy")
    print("Policy states: observe -> approach -> descend -> grasp -> lift -> place -> drop -> done")
    print("Press Ctrl+C to stop\n")
    
    # Debug first observations
    print("=== Observation Analysis ===")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, sample: {value[:min(5, len(value))]}")
        else:
            print(f"{key}: {value}")
    print("============================\n")
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"=== Episode {episode} ===")
            
            obs = env.reset()
            policy.reset()  # Reset policy state
            done = False
            step = 0
            total_reward = 0
            
            while not done:
                # Get action from predefined policy (only arm action now)
                action = policy.get_action(obs)
                
                # Take step with 7-DOF action
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Render
                env.render()
                
                # Print reward when non-zero or for debugging
                if reward > 0 or step % 100 == 0:
                    print(f"Step {step}: Reward = {reward:.3f} (Policy state: {policy.state})")
                
                # Check if episode is done
                if done:
                    print(f"Episode {episode} finished after {step} steps")
                    print(f"Total reward: {total_reward:.3f}")
                    break
                
                # Stop after reasonable time if policy gets stuck
                if step > 1000:
                    print(f"Episode {episode} stopped after {step} steps (timeout)")
                    print(f"Total reward: {total_reward:.3f}")
                    break
            
            # Brief pause between episodes
            import time
            time.sleep(3.0)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
