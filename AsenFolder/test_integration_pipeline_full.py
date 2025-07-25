#!/usr/bin/env python3
"""
test_integration_pipeline.py - Fixed version with proper camera observation handling

This version fixes the camera observation extraction issues that were causing
the VLA model to receive None images. Image processing logic is now in a separate module.
"""

import os  # Add this line
import time
import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import torch
from PIL import Image

# Import our image utilities
from image_utils import (
    extract_image_from_obs,
    analyze_all_camera_views,
    process_image_for_vla,
    analyze_vla_action,
    check_robot_state,
    create_camera_summary_plot,
    clear_debug_images,
    setup_debug_directories
)

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
        self.action_scaling = 1000.0  # Scaling factor for VLA actions
        self.max_action_magnitude = 10.0  # Maximum action magnitude for safety

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
                torch_dtype=torch.bfloat16,
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

    def get_vla_action(self, image, instruction="pick up the cube"):
        """
        Get action prediction from OpenVLA model
        Enhanced with image processing from separate module
        
        Args:
            image: RGB image as numpy array (H, W, 3) or PIL Image, or None
            instruction: Task instruction string
            
        Returns:
            numpy array: VLA action [x, y, z, roll, pitch, yaw, gripper]
        """
        # Handle None image case - return fallback action
        if image is None:
            print("⚠️  No image available - using fallback action")
            return np.array([0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0])
        
        try:
            # Simplified prompt format
            prompt = "pick up the cube"
            print(f"🎯 Using instruction: '{prompt}'")
            
            # Process image using our image utilities
            pil_image = process_image_for_vla(image, self.step_count)
            if pil_image is None:
                print("❌ Image processing failed")
                return np.array([0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0])
            
            # Verify the processed image dimensions and save a sample for verification
            if self.step_count % 40 == 0:  # Every 40 steps
                verify_dir = "debug_images/vla_verification"
                os.makedirs(verify_dir, exist_ok=True)
                verify_filename = f"{verify_dir}/vla_receives_step_{self.step_count:04d}.png"
                pil_image.save(verify_filename)
                print(f"🔍 VLA VERIFICATION: Saved what VLA actually receives: {verify_filename}")
            
            # Process inputs for model
            inputs = self.processor(prompt, pil_image).to(
                self.device, dtype=torch.bfloat16
            )
            
            print("After processing inputs, running inference...")

            # Use bridge_orig as the primary normalization (best for manipulation tasks)
            with torch.no_grad():
                action = self.vla_model.predict_action(
                    **inputs, 
                    unnorm_key="bridge_orig",  # Use BridgeData normalization
                    do_sample=False
                )
        
            print("VLA inference completed successfully")

            # Convert to numpy if needed
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            
            print(f"🎯 VLA Raw Action: {action}")
            print(f"   Action magnitude: {np.linalg.norm(action[:6]):.6f}")
            
            return action
            
        except Exception as e:
            print(f"❌ VLA inference error: {e}")
            import traceback
            traceback.print_exc()
            return np.array([0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0])
    
    def convert_vla_to_robosuite_action(self, vla_action):
        """
        Convert VLA output to robosuite IK controller input with enhanced debugging
        """
        # Extract position and orientation deltas
        position_delta = vla_action[:3]      # [dx, dy, dz]
        orientation_delta = vla_action[3:6]  # [droll, dpitch, dyaw]
        
        # Debug: Check if actions are too small
        pos_mag = np.linalg.norm(position_delta)
        rot_mag = np.linalg.norm(orientation_delta)
        
        if self.step_count % 20 == 0:
            print(f"🔧 Action Conversion Debug (Step {self.step_count}):")
            print(f"   Position magnitude: {pos_mag:.6f}")
            print(f"   Rotation magnitude: {rot_mag:.6f}")
            print(f"   Current scaling: {self.action_scaling}")
            print(f"   Max magnitude limit: {self.max_action_magnitude}")
        
        # Combine into 6D action for IK controller
        ik_action = np.concatenate([position_delta, orientation_delta])
        
        # Apply scaling
        ik_action_scaled = ik_action * self.action_scaling
        
        # Clip to safe ranges
        ik_action_final = np.clip(ik_action_scaled, -self.max_action_magnitude, self.max_action_magnitude)
        
        # Check if clipping is happening (which might indicate actions are too large)
        if self.step_count % 20 == 0:
            clipped = not np.allclose(ik_action_scaled, ik_action_final)
            if clipped:
                print(f"   ⚠️  Action was clipped! Scaled: {ik_action_scaled}")
                print(f"   ⚠️  After clipping: {ik_action_final}")
        
        return ik_action_final
    
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


def create_robosuite_env_with_cameras():
    """
    Create robosuite environment with proper camera configuration
    
    Returns:
        tuple: (env, has_cameras) where env is the raw environment (NOT wrapped) and has_cameras is bool
    """
    from robosuite.controllers.parts.controller_factory import load_part_controller_config
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
    
    controller_type = "IK_POSE"
    control_freq = 20
    
    print(f"   Loading {controller_type} controller configuration...")
    part_controller_config = load_part_controller_config(default_controller=controller_type)
    controller_config = refactor_composite_controller_config(
        part_controller_config, robot_type="Panda", arms=["right"]
    )
    
    # Set environment variables for rendering
    import os
    
    # Camera configuration
    camera_names = ["frontview", "agentview"]
    camera_height = 256
    camera_width = 256
    
    try:
        print("   Attempting to create environment with camera observations...")
        
        # Try GPU acceleration first
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['EGL_DEVICE_ID'] = '0'
        
        # Create raw environment WITHOUT GymWrapper to preserve camera observations
        raw_env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=True,
            render_camera="frontview",
            has_offscreen_renderer=True,        # Critical for camera observations
            control_freq=control_freq,
            horizon=1000,
            use_object_obs=True,
            use_camera_obs=True,                # Enable camera observations
            camera_names=camera_names,          # Specify camera names
            camera_heights=camera_height,       # Camera resolution
            camera_widths=camera_width,
            reward_shaping=True,
            render_gpu_device_id=0,
        )
        
        # Test if camera observations work with raw environment
        print("   Testing camera observations with raw environment...")
        raw_obs = raw_env.reset()
        
        # Check raw observation structure
        print(f"   Raw obs type: {type(raw_obs)}")
        if isinstance(raw_obs, dict):
            image_keys = [k for k in raw_obs.keys() if 'image' in k.lower()]
            print(f"   Raw env image keys: {image_keys}")
            
            # Test if any images are actually available
            has_valid_images = False
            for key in image_keys:
                img = raw_obs[key]
                if img is not None and hasattr(img, 'shape'):
                    print(f"   ✅ Valid image at {key}: {img.shape}")
                    has_valid_images = True
                else:
                    print(f"   ❌ Invalid image at {key}: {type(img)}")
            
            if has_valid_images:
                print("   ✅ Camera observations working with raw environment")
                print("   📋 NOT using GymWrapper to preserve camera observations")
                
                # Return raw environment to preserve camera observations
                return raw_env, True
            else:
                print("   ❌ No valid camera observations found")
                return raw_env, False
        else:
            print(f"   ❌ Raw observation is not a dict: {type(raw_obs)}")
            return raw_env, False
        
    except Exception as gpu_error:
        print(f"⚠️  GPU rendering failed: {type(gpu_error).__name__}: {gpu_error}")
        print("   Trying software rendering...")
        
        try:
            # Try software rendering
            os.environ['MUJOCO_GL'] = 'osmesa'
            
            raw_env = suite.make(
                env_name="PickPlace",
                robots="Panda",
                controller_configs=controller_config,
                has_renderer=True,
                render_camera="frontview",
                has_offscreen_renderer=True,
                control_freq=control_freq,
                horizon=1000,
                use_object_obs=True,
                use_camera_obs=True,
                camera_names=camera_names,
                camera_heights=camera_height,
                camera_widths=camera_width,
                reward_shaping=True,
            )
            
            # Test software rendering
            raw_obs = raw_env.reset()
            if isinstance(raw_obs, dict):
                image_keys = [k for k in raw_obs.keys() if 'image' in k.lower()]
                if image_keys and any(raw_obs[k] is not None for k in image_keys):
                    print("   ✅ Software rendering working")
                    return raw_env, True
            
            print("   ⚠️  Software rendering available but no camera observations")
            return raw_env, False
            
        except Exception as software_error:
            print(f"⚠️  Software rendering failed: {type(software_error).__name__}: {software_error}")
            print("   Using no-rendering mode...")
            
            # Fallback: no rendering
            raw_env = suite.make(
                env_name="PickPlace",
                robots="Panda",
                controller_configs=controller_config,
                has_renderer=False,
                has_offscreen_renderer=False,
                control_freq=control_freq,
                horizon=1000,
                use_object_obs=True,
                use_camera_obs=False,
                reward_shaping=True,
            )
            
            return raw_env, False


def test_integration():
    """
    Test the complete integration pipeline with proper camera observation handling
    """
    print("🧪 Testing OpenVLA + Robosuite Integration Pipeline (Fixed Version)")
    print("=" * 70)
    
    # Setup debug directories and clear old images
    clear_debug_images()
    setup_debug_directories()
    
    # Configuration
    max_steps_per_episode = 2000
    max_episodes = 2
    
    try:
        # Step 1: Initialize VLA controller
        print("\n🤖 Step 1: Initializing VLA Controller")
        vla_controller = OpenVLAController()

        # Step 2: Create robosuite environment with proper camera setup
        print("\n🏗️  Step 2: Creating Robosuite Environment with Camera Support")
        env, has_cameras = create_robosuite_env_with_cameras()
        
        print(f"✅ Environment created successfully")
        print(f"   Camera observations available: {has_cameras}")
        
        # Get robot info
        robot = env.robots[0]  # Raw environment access
        arm = robot.arms[0]
        
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
            
            while not done and step < max_steps_per_episode:
                # Analyze all camera views periodically using image utilities
                analyze_all_camera_views(obs, step)
                
                # Extract camera image using image utilities
                image = extract_image_from_obs(obs, step)
                
                # Get VLA action (handles None image gracefully)
                if image is not None and step % 10 == 0:
                    print(f"📷 Using camera image: {image.shape} {image.dtype}")
                elif image is None and step % 10 == 0:
                    print("📷 No camera image - using fallback VLA action")
                
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
                obs, reward, done, info = env.step(action)  # Raw env returns (obs, reward, done, info)
                total_reward += reward
                step += 1
                
                # Update controller
                vla_controller.update_step_count()
                
                # Print progress
                if reward > 0:
                    print(f"✅ Step {step}: Reward = {reward:.3f} (Total: {total_reward:.3f})")
                elif step % 25 == 0:
                    print(f"📊 Step {step}: Reward = {reward:.3f} (Total: {total_reward:.3f})")
                
                # Check robot state using image utilities
                check_robot_state(obs, step)
                
                # Create summary visualization every 100 steps
                if step % 100 == 0:
                    create_camera_summary_plot()
                
                # Small delay for visualization
                time.sleep(0.05)
            
            print(f"🏁 Test Episode {episode + 1} completed:")
            print(f"   Steps: {step}")
            print(f"   Total Reward: {total_reward:.3f}")
            print(f"   Success: {'Yes' if total_reward > 0.5 else 'No'}")
            print(f"   Camera available: {has_cameras}")
            
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