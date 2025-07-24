#!/usr/bin/env python3
"""
test_integration_pipeline.py - Test version with dummy VLA outputs

This is a simpler version that tests the integration pipeline without requiring
the full OpenVLA model. Uses dummy VLA outputs for testing the control flow.
"""

import time
import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

class DummyVLAController:
    """
    Dummy VLA controller for testing the integration pipeline
    """
    
    def __init__(self, controller_type="IK_POSE"):
        self.controller_type = controller_type
        self.step_count = 0
        self.action_scaling = 0.5  # Reduced for safety during testing
        self.max_action_magnitude = 0.02  # Smaller movements for testing
        
        print(f"🧪 Initializing Dummy VLA Controller")
        print(f"   Controller: {controller_type}")
        print("   Using dummy VLA outputs for testing")
    
    def get_vla_action(self, image, instruction="pick up the cube"):
        """
        Return dummy VLA action for testing
        Uses your example output with some variation
        """
        # Base action (your example)
        base_action = np.array([-0.00848473, 0.01025995, -0.00102072, 
                               0.01135582, 0.00323859, -0.0065139, 0.0000])
        
        # Add small random variation to simulate different actions
        noise = np.random.normal(0, 0.001, 7)  # Small random noise
        dummy_action = base_action + noise
        
        # Simulate different phases based on step count
        if self.step_count < 50:
            # Approach phase - move towards object
            dummy_action[0] = 0.01  # Move forward
            dummy_action[2] = -0.005  # Move down slightly
            dummy_action[6] = 1.0   # Keep gripper open
        elif self.step_count < 100:
            # Grasp phase - close gripper
            dummy_action[:6] *= 0.1  # Smaller movements
            dummy_action[6] = -1.0   # Close gripper
        elif self.step_count < 150:
            # Lift phase - move up
            dummy_action[2] = 0.01   # Move up
            dummy_action[6] = -1.0   # Keep gripper closed
        else:
            # Move to target phase
            dummy_action[0] = -0.01  # Move backward
            dummy_action[1] = 0.01   # Move sideways
            dummy_action[6] = -1.0   # Keep gripper closed
        
        return dummy_action
    
    def convert_vla_to_robosuite_action(self, vla_action):
        """Convert VLA output to robosuite IK controller input"""
        # Extract position and orientation deltas
        position_delta = vla_action[:3]      # [dx, dy, dz]
        orientation_delta = vla_action[3:6]  # [droll, dpitch, dyaw]
        
        # Combine into 6D action for IK controller
        ik_action = np.concatenate([position_delta, orientation_delta])
        
        # Apply scaling
        ik_action = ik_action * self.action_scaling
        
        # Clip to safe ranges
        ik_action = np.clip(ik_action, -self.max_action_magnitude, self.max_action_magnitude)
        
        return ik_action
    
    def get_gripper_action(self, vla_action, current_gripper_state=0.04):
        """Extract gripper command from VLA action"""
        if len(vla_action) > 6:
            gripper_command = vla_action[6]
            
            # Simple threshold-based gripper control
            if gripper_command > 0.5:
                return 0.04  # Open gripper
            elif gripper_command < -0.5:
                return 0.0   # Close gripper
            else:
                return current_gripper_state  # Maintain current state
        else:
            return current_gripper_state
    
    def debug_print_action(self, vla_action, robosuite_action):
        """Print debug information about actions"""
        if self.step_count % 20 == 0:  # Print every 20 steps
            print(f"🎯 Step {self.step_count} - Action Analysis:")
            print(f"   VLA Raw:          {vla_action}")
            print(f"   Position Delta:   [{vla_action[0]:.4f}, {vla_action[1]:.4f}, {vla_action[2]:.4f}]")
            print(f"   Orientation Delta:[{vla_action[3]:.4f}, {vla_action[4]:.4f}, {vla_action[5]:.4f}]")
            if len(vla_action) > 6:
                gripper_state = "OPEN" if vla_action[6] > 0.5 else "CLOSE" if vla_action[6] < -0.5 else "HOLD"
                print(f"   Gripper:          {vla_action[6]:.4f} ({gripper_state})")
            print(f"   Robosuite Action: {robosuite_action}")
    
    def update_step_count(self):
        """Update step counter"""
        self.step_count += 1


def test_integration():
    """
    Test the complete integration pipeline with dummy VLA outputs
    """
    print("🧪 Testing OpenVLA + Robosuite Integration Pipeline")
    print("=" * 60)
    
    # Configuration
    controller_type = "IK_POSE"  # Start with IK_POSE
    control_freq = 20
    max_steps_per_episode = 200  # Shorter episodes for testing
    max_episodes = 2
    
    try:
        # Step 1: Initialize dummy VLA controller
        print("\n🤖 Step 1: Initializing Dummy VLA Controller")
        vla_controller = DummyVLAController(controller_type=controller_type)
        
        # Step 2: Create robosuite environment
        print("\n🏗️  Step 2: Creating Robosuite Environment")
        
        # Load controller configuration (IK_POSE is a part controller, not composite)
        from robosuite.controllers.parts.controller_factory import load_part_controller_config
        from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
        
        print(f"   Loading {controller_type} controller configuration...")
        part_controller_config = load_part_controller_config(default_controller=controller_type)
        controller_config = refactor_composite_controller_config(
            part_controller_config, robot_type="Panda", arms=["right"]
        )
        
        # Create environment
        raw_env = suite.make(
            env_name="PickPlace",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=True,
            render_camera="frontview",
            has_offscreen_renderer=True,        # Enable offscreen renderer for camera obs
            control_freq=control_freq,
            horizon=1000,
            use_object_obs=True,
            use_camera_obs=True,                # Camera observations for VLA
            camera_names=["frontview", "agentview"],
            camera_heights=256,
            camera_widths=256,
            reward_shaping=True,
        )
        
        env = GymWrapper(raw_env)
        robot = env.env.robots[0]
        arm = robot.arms[0]
        
        print("✅ Environment created successfully")
        print("📊 Action Space Information:")
        robot.print_action_info()
        
        # Step 3: Run test episodes
        print("\n🎯 Step 3: Running Test Episodes")
        
        for episode in range(max_episodes):
            print(f"\n=== Test Episode {episode + 1}/{max_episodes} ===")
            
            # Reset environment
            obs = env.reset()
            print("🔄 Environment reset")
            
            done = False
            step = 0
            total_reward = 0.0
            current_gripper = 0.04  # Start with open gripper
            
            # Debug: Print observation info
            print(f"📊 Observation type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"📊 Observation keys: {list(obs.keys())}")
                image_keys = [k for k in obs.keys() if "image" in k]
                print(f"📷 Available cameras: {image_keys}")
            
            while not done and step < max_steps_per_episode:
                # Get camera image 
                image = None
                if isinstance(obs, dict):
                    if "frontview_image" in obs:
                        image = obs["frontview_image"]
                    elif "agentview_image" in obs:
                        image = obs["agentview_image"]
                    else:
                        # Find any image observation
                        image_keys = [k for k in obs.keys() if "image" in k]
                        if image_keys:
                            image = obs[image_keys[0]]
                            print(f"📷 Using camera: {image_keys[0]}")
                else:
                    print(f"⚠️  Unexpected obs type: {type(obs)}")
                
                # Get dummy VLA action
                vla_action = vla_controller.get_vla_action(image)
                
                # Convert to robosuite action
                ik_action = vla_controller.convert_vla_to_robosuite_action(vla_action)
                gripper_action = vla_controller.get_gripper_action(vla_action, current_gripper)
                current_gripper = gripper_action
                
                # Debug output
                vla_controller.debug_print_action(vla_action, ik_action)
                
                # Create full action vector
                action = robot.create_action_vector({
                    arm: ik_action,
                    f"{arm}_gripper": np.array([gripper_action])
                })
                
                # Take environment step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
                
                # Update controller
                vla_controller.update_step_count()
                
                # Render
                env.render()
                
                # Print progress
                if reward > 0:
                    print(f"✅ Step {step}: Reward = {reward:.3f} (Total: {total_reward:.3f})")
                elif step % 25 == 0:
                    print(f"📊 Step {step}: Reward = {reward:.3f} (Total: {total_reward:.3f})")
                
                # Print end-effector position occasionally
                if step % 50 == 0 and "robot0_eef_pos" in obs:
                    ee_pos = obs["robot0_eef_pos"]
                    print(f"   EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                
                # Small delay for visualization
                time.sleep(0.05)  # Slightly slower for testing
            
            print(f"🏁 Test Episode {episode + 1} completed:")
            print(f"   Steps: {step}")
            print(f"   Total Reward: {total_reward:.3f}")
            print(f"   Success: {'Yes' if total_reward > 0.5 else 'No'}")
            
            # Reset step count for next episode
            vla_controller.step_count = 0
            
            # Brief pause between episodes
            time.sleep(2.0)
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if 'env' in locals():
            env.close()
        print("✅ Test finished")


if __name__ == "__main__":
    test_integration()
