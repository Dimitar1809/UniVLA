#!/usr/bin/env python3
"""
vla_integration_example.py - Complete OpenVLA + Robosuite Integration Pipeline

This script combines:
- OpenVLA model loading and inference
- Robosuite IK controller setup  
- Vision-based robot control
- Complete pick-and-place task execution

Based on GPU_INTEGRATION_GUIDE.md for proper robosuite IK integration.
"""

import time
import numpy as np
import torch
from PIL import Image
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

class OpenVLAController:
    """
    Complete OpenVLA controller that integrates VLA model with robosuite IK control
    """
    
    def __init__(self, model_name="openvla/openvla-7b", controller_type="IK_POSE"):
        """
        Initialize OpenVLA controller
        
        Args:
            model_name: HuggingFace model name for OpenVLA
            controller_type: Robosuite controller type ("IK_POSE", "OSC_POSE", "WholeBodyIK")
        """
        self.model_name = model_name
        self.controller_type = controller_type
        self.step_count = 0
        
        # Initialize model components
        self.processor = None
        self.vla_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Control parameters
        self.action_scaling = 1.0  # Scaling factor for VLA actions
        self.max_action_magnitude = 0.05  # Maximum action magnitude for safety
        
        print(f"🤖 Initializing OpenVLA Controller")
        print(f"   Model: {model_name}")
        print(f"   Controller: {controller_type}")
        print(f"   Device: {self.device}")
        
        # Load VLA model
        self._load_vla_model()
    
    def _load_vla_model(self):
        """Load OpenVLA model and processor"""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            print("📦 Loading OpenVLA processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            print("📦 Loading OpenVLA model...")
            self.vla_model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                attn_implementation="flash_attention_2",  # Use flash attention if available
                torch_dtype=torch.bfloat16,               # Use bfloat16 for efficiency
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
            
            print(f"✅ OpenVLA model loaded successfully on {self.device}")
            
        except ImportError as e:
            print(f"❌ Failed to import transformers: {e}")
            print("   Install with: pip install transformers torch")
            raise
        except Exception as e:
            print(f"❌ Failed to load OpenVLA model: {e}")
            raise
    
    def get_vla_action(self, image, instruction="pick up the cube and place it in the target location"):
        """
        Get action prediction from OpenVLA model
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Task instruction string
            
        Returns:
            numpy array: VLA action [x, y, z, roll, pitch, yaw, gripper]
        """
        try:
            # Format prompt for OpenVLA
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
            
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                # Ensure proper format
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Process inputs for model
            inputs = self.processor(prompt, pil_image).to(
                self.device, dtype=torch.bfloat16
            )
            
            # Get action prediction
            with torch.no_grad():
                action = self.vla_model.predict_action(
                    **inputs, 
                    unnorm_key="bridge_orig",  # Use BridgeData normalization
                    do_sample=False
                )
            
            # Convert to numpy if needed
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            
            return action
            
        except Exception as e:
            print(f"❌ VLA inference error: {e}")
            # Return fallback action (your example output)
            return np.array([-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000])
    
    def convert_vla_to_robosuite_action(self, vla_action):
        """
        Convert VLA output to robosuite IK controller input
        
        VLA Output: [x, y, z, roll, pitch, yaw, gripper]
        IK Input:   [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z]
        
        Args:
            vla_action: VLA model output (7-element array)
            
        Returns:
            numpy array: Robosuite action for IK controller (6-element array)
        """
        # Extract position and orientation deltas
        position_delta = vla_action[:3]      # [dx, dy, dz]
        orientation_delta = vla_action[3:6]  # [droll, dpitch, dyaw]
        # Note: vla_action[6] is gripper (handled separately)
        
        # Combine into 6D action for IK controller
        ik_action = np.concatenate([position_delta, orientation_delta])
        
        # Apply scaling
        ik_action = ik_action * self.action_scaling
        
        # Clip to safe ranges
        ik_action = np.clip(ik_action, -self.max_action_magnitude, self.max_action_magnitude)
        
        return ik_action
    
    def get_gripper_action(self, vla_action, current_gripper_state=0.04):
        """
        Extract gripper command from VLA action
        
        Args:
            vla_action: VLA model output
            current_gripper_state: Current gripper opening (0.0 = closed, 0.04 = open)
            
        Returns:
            float: Gripper command
        """
        if len(vla_action) > 6:
            gripper_command = vla_action[6]
            
            # Simple threshold-based gripper control
            # Adjust thresholds based on your VLA model's output range
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
            print(f"🎯 Step {self.step_count} - VLA Action Analysis:")
            print(f"   VLA Raw:        {vla_action}")
            print(f"   Position:       [{vla_action[0]:.4f}, {vla_action[1]:.4f}, {vla_action[2]:.4f}]")
            print(f"   Orientation:    [{vla_action[3]:.4f}, {vla_action[4]:.4f}, {vla_action[5]:.4f}]")
            if len(vla_action) > 6:
                print(f"   Gripper:        {vla_action[6]:.4f}")
            print(f"   Robosuite Action: {robosuite_action}")
    
    def update_step_count(self):
        """Update step counter"""
        self.step_count += 1


class VLAPickPlaceEnvironment:
    """
    Environment wrapper for VLA-controlled pick and place task
    """
    
    def __init__(self, controller_type="IK_POSE", control_freq=20):
        """
        Initialize environment with proper IK controller
        
        Args:
            controller_type: Type of robosuite controller to use
            control_freq: Control frequency in Hz
        """
        self.controller_type = controller_type
        self.control_freq = control_freq
        self.env = None
        self.robot = None
        self.arm = None
        
        print(f"🏗️  Setting up robosuite environment...")
        print(f"   Controller: {controller_type}")
        print(f"   Frequency: {control_freq} Hz")
        
        self._create_environment()
    
    def _create_environment(self):
        """Create robosuite environment with IK controller"""
        # Load IK controller configuration properly
        from robosuite.controllers.parts.controller_factory import load_part_controller_config
        from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
        
        print(f"   Loading {self.controller_type} controller configuration...")
        part_controller_config = load_part_controller_config(default_controller=self.controller_type)
        controller_config = refactor_composite_controller_config(
            part_controller_config, robot_type="Panda", arms=["right"]
        )
        
        # Create environment with IK control
        raw_env = suite.make(
            env_name="PickPlace",               # Pick and place task
            robots="Panda",                     # Panda robot
            controller_configs=controller_config, # IK controller
            has_renderer=True,                  # Enable visualization
            render_camera="frontview",          # Camera for rendering
            has_offscreen_renderer=True,        # Enable offscreen renderer for camera obs
            control_freq=self.control_freq,     # Control frequency
            horizon=1000,                       # Episode length
            use_object_obs=True,                # Object state observations
            use_camera_obs=True,                # Camera observations for VLA
            camera_names=["frontview", "agentview"], # Multiple cameras
            camera_heights=256,                 # Image height
            camera_widths=256,                  # Image width
            reward_shaping=True,                # Shaped rewards
        )
        
        # Wrap environment
        self.env = GymWrapper(raw_env)
        self.robot = self.env.env.robots[0]
        self.arm = self.robot.arms[0]
        
        print("✅ Environment created successfully")
        
        # Print action space information
        print("📊 Action Space Information:")
        self.robot.print_action_info()
    
    def reset(self):
        """Reset environment and return initial observation"""
        obs = self.env.reset()
        print("🔄 Environment reset")
        return obs
    
    def step(self, ik_action, gripper_action):
        """
        Take environment step with IK action and gripper command
        
        Args:
            ik_action: 6D action for IK controller [dx, dy, dz, dax, day, daz]
            gripper_action: Gripper command (float)
            
        Returns:
            observation, reward, done, info
        """
        # Create full action vector
        action = self.robot.create_action_vector({
            self.arm: ik_action,
            f"{self.arm}_gripper": np.array([gripper_action])
        })
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        return obs, reward, done, info
    
    def render(self):
        """Render environment"""
        self.env.render()
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def get_camera_image(self, obs, camera_name="frontview_image"):
        """Extract camera image from observation"""
        if camera_name in obs:
            return obs[camera_name]
        else:
            # Try to find any camera observation
            camera_keys = [k for k in obs.keys() if "image" in k]
            if camera_keys:
                print(f"⚠️  Camera '{camera_name}' not found, using '{camera_keys[0]}'")
                return obs[camera_keys[0]]
            else:
                print("❌ No camera observations found")

def main():
    """
    Main function that runs the complete VLA integration pipeline
    """
    print("🚀 OpenVLA + Robosuite Integration Pipeline")
    print("=" * 60)
    
    # Configuration
    model_name = "openvla/openvla-7b"
    controller_type = "IK_POSE"  # or "OSC_POSE", "WholeBodyIK"
    control_freq = 20
    max_steps_per_episode = 500
    max_episodes = 5
    
    try:
        # Step 1: Initialize OpenVLA controller
        print("\n🤖 Step 1: Initializing OpenVLA Controller")
        vla_controller = OpenVLAController(
            model_name=model_name,
            controller_type=controller_type
        )
        
        # Step 2: Create robosuite environment
        print("\n🏗️  Step 2: Creating Robosuite Environment")
        env = VLAPickPlaceEnvironment(
            controller_type=controller_type,
            control_freq=control_freq
        )
        
        # Step 3: Run VLA-controlled episodes
        print("\n🎯 Step 3: Running VLA-Controlled Episodes")
        print("Press Ctrl+C to stop")
        
        for episode in range(max_episodes):
            print(f"\n=== Episode {episode + 1}/{max_episodes} ===")
            
            # Reset environment
            obs = env.reset()
            done = False
            step = 0
            total_reward = 0.0
            current_gripper = 0.04  # Start with open gripper
            
            while not done and step < max_steps_per_episode:
                # Extract camera image
                image = env.get_camera_image(obs)
                if image is None:
                    print("❌ No camera image available, skipping step")
                    break
                
                # Get VLA action
                vla_action = vla_controller.get_vla_action(image)
                
                # Convert to robosuite action
                ik_action = vla_controller.convert_vla_to_robosuite_action(vla_action)
                gripper_action = vla_controller.get_gripper_action(vla_action, current_gripper)
                current_gripper = gripper_action
                
                # Debug output
                vla_controller.debug_print_action(vla_action, ik_action)
                
                # Take environment step
                obs, reward, done, info = env.step(ik_action, gripper_action)
                total_reward += reward
                step += 1
                
                # Update controller
                vla_controller.update_step_count()
                
                # Render
                env.render()
                
                # Print progress
                if reward > 0:
                    print(f"✅ Step {step}: Reward = {reward:.3f} (Total: {total_reward:.3f})")
                elif step % 50 == 0:
                    print(f"📊 Step {step}: Reward = {reward:.3f} (Total: {total_reward:.3f})")
                
                # Print end-effector position occasionally
                if step % 100 == 0 and "robot0_eef_pos" in obs:
                    ee_pos = obs["robot0_eef_pos"]
                    print(f"   EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                
                # Small delay for visualization
                time.sleep(0.01)
            
            print(f"🏁 Episode {episode + 1} completed:")
            print(f"   Steps: {step}")
            print(f"   Total Reward: {total_reward:.3f}")
            print(f"   Success: {'Yes' if total_reward > 0.5 else 'No'}")
            
            # Brief pause between episodes
            time.sleep(2.0)
    
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if 'env' in locals():
            env.close()
        print("✅ Pipeline finished")


if __name__ == "__main__":
    main()
        
        # PLACEHOLDER: Your actual model output
        # Based on your example: [-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000]
        vla_output = np.array([-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000])
        
        return vla_output
    
    def _prepare_observation_for_vla(self, obs):
        """
        Convert robosuite observation to format expected by VLA model
        """
        # Get RGB image from robosuite
        if "frontview_image" in obs:
            rgb_image = obs["frontview_image"]
        elif "agentview_image" in obs:
            rgb_image = obs["agentview_image"] 
        else:
            # Fallback: create dummy image
            rgb_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Get proprioception (robot state)
        robot_state = obs.get("robot0_proprio-state", np.zeros(32))
        ee_pos = obs.get("robot0_eef_pos", np.zeros(3))
        
        # Combine into format your VLA model expects
        vla_observation = {
            'image': rgb_image,
            'proprio': np.concatenate([robot_state, ee_pos]),
            'ee_pos': ee_pos,
        }
        
        return vla_observation

def main():
    """
    Example of how to use your VLA model with robosuite
    """
    print("🚀 VLA Model Integration Example")
    
    # TODO: Load your actual VLA model here
    # vla_model = load_your_vla_model()
    vla_model = None  # Placeholder
    
    # Create environment with camera observations for VLA
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=True,
        render_camera="frontview",
        has_offscreen_renderer=False,
        use_camera_obs=True,  # Important: VLA needs visual input
        camera_names=["frontview"],  # Specify camera for VLA
        camera_heights=224,   # Standard size for vision models
        camera_widths=224,
        control_freq=20,
        horizon=500,
        use_object_obs=True,
        reward_shaping=True,
    )
    
    # Create your VLA policy
    policy = YourVLAModelPolicy(vla_model=vla_model)
    
    print("🎯 Starting VLA-controlled robot...")
    
    try:
        obs = env.reset()
        done = False
        step = 0
        
        while not done and step < 100:  # Short demo
            # Get action from VLA model
            action = policy.get_action(obs)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            step += 1
            
            # Render
            env.render()
            
            if step % 20 == 0:
                print(f"Step {step}, Reward: {reward:.3f}")
        
        print(f"✅ Demo completed after {step} steps")
        
    except KeyboardInterrupt:
        print("🛑 Demo interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    main()
