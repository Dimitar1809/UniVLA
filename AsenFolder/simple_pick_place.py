#!/usr/bin/env python3
"""
simple_pick_place.py - A very simple, direct approach to pick and place

This script focuses on:
1. One object at a time (milk)
2. Simple, direct movements
3. Understanding what the robot can actually do
"""

import numpy as np
import robosuite as suite

class VerySimplePolicy:
    """
    An extremely simple policy that just tries to move the end-effector
    to specific target positions step by step
    """
    def __init__(self):
        self.step_count = 0
        self.target_positions = [
            np.array([0.0, -0.2, 1.0]),   # Move above object
            np.array([0.0, -0.2, 0.89]),  # Move down to object  
            np.array([0.0, -0.2, 1.0]),   # Lift up
            np.array([0.0, 0.2, 1.0]),    # Move to target area
            np.array([0.0, 0.2, 0.89]),   # Lower down
        ]
        self.current_target = 0
        self.steps_at_target = 0
        
    def get_action(self, obs):
        """
        Very simple action: try to move end-effector toward current target
        """
        # Get current end-effector position
        ee_pos = obs.get("robot0_eef_pos", np.array([0.0, 0.0, 1.0]))
        
        # Print status every 20 steps
        if self.step_count % 20 == 0:
            print(f"Step {self.step_count}: Target {self.current_target}")
            print(f"  EE: {ee_pos}")
            print(f"  Target: {self.target_positions[self.current_target]}")
            print(f"  Error: {np.linalg.norm(ee_pos - self.target_positions[self.current_target]):.3f}")
        
        # Check if we've reached current target
        if self.current_target < len(self.target_positions):
            target = self.target_positions[self.current_target]
            error = target - ee_pos
            distance = np.linalg.norm(error)
            
            # If close enough or spent enough time, move to next target
            if distance < 0.05 or self.steps_at_target > 50:
                self.current_target += 1
                self.steps_at_target = 0
                if self.current_target < len(self.target_positions):
                    print(f"*** Moving to target {self.current_target} ***")
            else:
                self.steps_at_target += 1
        
        # Generate action based on current target
        action = np.zeros(7)
        if self.current_target < len(self.target_positions):
            target = self.target_positions[self.current_target]
            error = target - ee_pos
            
            # Very simple control - just proportional movement
            # Scale errors to reasonable joint velocities
            action[0] = np.clip(error[1] * 1.0, -0.3, 0.3)   # Base joint for Y
            action[1] = np.clip(error[2] * 0.8, -0.3, 0.3)   # Shoulder for Z
            action[2] = np.clip(error[0] * 0.8, -0.3, 0.3)   # Elbow for X
            action[3] = np.clip(-error[2] * 0.2, -0.1, 0.1)  # Small wrist adjustments
            action[4] = np.clip(error[0] * 0.1, -0.1, 0.1)   
            action[5] = np.clip(error[1] * 0.1, -0.1, 0.1)   
            action[6] = 0.0  # Keep end-effector orientation stable
        
        self.step_count += 1
        return action
    
    def reset(self):
        self.step_count = 0
        self.current_target = 0
        self.steps_at_target = 0

def main():
    print("Creating simple PickPlace environment...")
    
    # Create environment
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=True,
        render_camera="frontview",
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=1000,  # Longer episode for debugging
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
    )
    
    print("Environment created!")
    print("This policy will try to:")
    print("1. Move above an object")
    print("2. Move down to touch it")
    print("3. Lift up")
    print("4. Move to target area") 
    print("5. Lower down")
    print("Note: Gripper control might not be available in action space")
    print()
    
    # Create simple policy
    policy = VerySimplePolicy()
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"=== Episode {episode} ===")
            
            obs = env.reset()
            policy.reset()
            done = False
            step = 0
            total_reward = 0
            
            # Show available objects
            print("Available objects:")
            for obj in ["Milk", "Bread", "Cereal", "Can"]:
                pos_key = f"{obj}_pos"
                if pos_key in obs:
                    print(f"  {obj}: {obs[pos_key]}")
            print()
            
            while not done and step < 500:  # Limit steps for debugging
                # Get action
                action = policy.get_action(obs)
                
                # Take step
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Render
                env.render()
                
                # Print rewards occasionally
                if reward > 0.001 or step % 100 == 0:
                    print(f"Step {step}: Reward = {reward:.3f}")
                
                if done:
                    print(f"Episode {episode} finished after {step} steps")
                    print(f"Total reward: {total_reward:.3f}")
                    break
            
            if step >= 500:
                print(f"Episode {episode} timed out after {step} steps")
                print(f"Total reward: {total_reward:.3f}")
            
            # Pause between episodes
            import time
            time.sleep(3.0)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
