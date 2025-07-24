#!/usr/bin/env python3
"""
openvla_integration_complete.py - Complete OpenVLA Integration Examples

This file shows three different ways to integrate OpenVLA with robosuite:
1. Direct Model Loading (HuggingFace)
2. REST API Client (Server-Client)
3. Placeholder for custom integration

Choose the approach that best fits your setup!
"""

import numpy as np
import robosuite as suite
import torch

# =====================================================================
# OPTION 1: Direct Model Loading Integration
# =====================================================================

class DirectOpenVLAIntegration:
    """
    Load OpenVLA model directly using HuggingFace transformers
    
    Requirements:
    pip install transformers torch torchvision timm tokenizers
    pip install flash-attn  # Optional but recommended
    """
    
    def __init__(self, model_name="openvla/openvla-7b"):
        print(f"🤖 Loading OpenVLA model: {model_name}")
        self.model_name = model_name
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        # Load model and processor
        self._load_model()
    
    def _load_model(self):
        """Load OpenVLA model using HuggingFace AutoClasses"""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Load VLA model
            self.vla = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                attn_implementation="flash_attention_2",  # Optional: Requires flash_attn
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
            
            print(f"✅ OpenVLA model loaded successfully on {self.device}")
            
        except ImportError as e:
            print(f"❌ Failed to import transformers: {e}")
            print("   Install with: pip install transformers torch")
            self.vla = None
            self.processor = None
        except Exception as e:
            print(f"❌ Failed to load OpenVLA model: {e}")
            self.vla = None
            self.processor = None
    
    def predict_action(self, image, instruction="pick up the cube and place it in the target location"):
        """
        Predict action using direct model inference
        
        Args:
            image: np.ndarray (H, W, 3) RGB image
            instruction: str task description
            
        Returns:
            np.ndarray: 7-DoF action [x, y, z, roll, pitch, yaw, gripper]
        """
        if self.vla is None:
            print("⚠️  Model not loaded, returning dummy action")
            return np.array([-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000])
        
        try:
            from PIL import Image as PILImage
            
            # Format prompt for OpenVLA
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
            
            # Process inputs
            inputs = self.processor(prompt, PILImage.fromarray(image).convert("RGB")).to(
                self.device, dtype=torch.bfloat16
            )
            
            # Get action prediction
            action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            return action
            
        except Exception as e:
            print(f"❌ VLA inference error: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# =====================================================================
# OPTION 2: REST API Client Integration  
# =====================================================================

class APIOpenVLAIntegration:
    """
    Use OpenVLA via REST API server
    
    Requirements:
    1. Start OpenVLA server:
       python vla-scripts/deploy.py --openvla_path openvla/openvla-7b
    2. Install client dependencies:
       pip install requests json-numpy
    """
    
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
        self.session = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup HTTP client for API communication"""
        try:
            import requests
            import json_numpy
            json_numpy.patch()  # Enable numpy serialization
            
            self.session = requests.Session()
            
            # Test server connection
            try:
                response = self.session.get(f"{self.server_url}/")
                print(f"✅ Connected to OpenVLA server at {self.server_url}")
            except:
                print(f"⚠️  Could not connect to server at {self.server_url}")
                print(f"   Make sure to start the server with:")
                print(f"   python vla-scripts/deploy.py --openvla_path openvla/openvla-7b")
                
        except ImportError as e:
            print(f"❌ Failed to import API dependencies: {e}")
            print("   Install with: pip install requests json-numpy")
            self.session = None
    
    def predict_action(self, image, instruction="pick up the cube and place it in the target location"):
        """
        Predict action using REST API
        
        Args:
            image: np.ndarray (H, W, 3) RGB image
            instruction: str task description
            
        Returns:
            np.ndarray: 7-DoF action [x, y, z, roll, pitch, yaw, gripper]
        """
        if self.session is None:
            print("⚠️  API client not setup, returning dummy action")
            return np.array([-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000])
        
        try:
            # Prepare payload
            payload = {
                "image": image,
                "instruction": instruction,
                "unnorm_key": "bridge_orig"  # For BridgeData V2 unnormalization
            }
            
            # Make API request
            response = self.session.post(f"{self.server_url}/act", json=payload)
            result = response.json()
            
            # Parse response
            if isinstance(result, dict) and "action" in result:
                return np.array(result["action"])
            else:
                return np.array(result)  # Direct action array
                
        except Exception as e:
            print(f"❌ API request error: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# =====================================================================
# OPTION 3: Custom Integration Template
# =====================================================================

class CustomVLAIntegration:
    """
    Template for your custom VLA model integration
    Replace this with your specific model loading and inference code
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self._load_your_model()
    
    def _load_your_model(self):
        """
        Replace this with your custom model loading code
        """
        print("🔧 Custom VLA model integration - implement your loading logic here")
        # Example:
        # self.model = YourVLAModel.load(self.model_path)
        pass
    
    def predict_action(self, image, instruction="pick up the cube and place it in the target location"):
        """
        Replace this with your custom inference code
        
        Args:
            image: np.ndarray (H, W, 3) RGB image  
            instruction: str task description
            
        Returns:
            np.ndarray: 7-DoF action [x, y, z, roll, pitch, yaw, gripper]
        """
        # Replace with your model's inference
        # action = self.model.predict(image, instruction)
        
        # For now, return the example action you provided
        return np.array([-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000])

# =====================================================================
# Unified VLA Policy for Robosuite
# =====================================================================

class OpenVLAPolicy:
    """
    Unified VLA policy that works with any of the integration methods above
    """
    
    def __init__(self, integration_type="direct", **kwargs):
        """
        Initialize VLA policy with chosen integration method
        
        Args:
            integration_type: "direct", "api", or "custom"
            **kwargs: Arguments passed to the integration class
        """
        self.integration_type = integration_type
        self.step_count = 0
        
        # Choose integration method
        if integration_type == "direct":
            self.vla_integration = DirectOpenVLAIntegration(**kwargs)
        elif integration_type == "api":
            self.vla_integration = APIOpenVLAIntegration(**kwargs)
        elif integration_type == "custom":
            self.vla_integration = CustomVLAIntegration(**kwargs)
        else:
            raise ValueError(f"Unknown integration type: {integration_type}")
        
        print(f"🎯 OpenVLA Policy initialized with {integration_type} integration")
    
    def get_action(self, obs):
        """
        Get action from VLA model based on current observation
        """
        # Prepare observation for VLA model
        image, instruction = self._prepare_observation_for_vla(obs)
        
        # Get VLA prediction
        vla_action = self.vla_integration.predict_action(image, instruction)
        
        # Print VLA output for debugging
        if self.step_count % 20 == 0:
            print(f"🎯 VLA Model Output (Step {self.step_count}):")
            print(f"  Raw action: {vla_action}")
            self._print_action_breakdown(vla_action)
        
        # Convert VLA output to robosuite action format
        robosuite_action = self._convert_vla_to_robosuite(vla_action, obs)
        
        self.step_count += 1
        return robosuite_action
    
    def _prepare_observation_for_vla(self, obs):
        """
        Prepare robosuite observation for OpenVLA model
        """
        # Extract camera observation (RGB image)
        if "frontview_image" in obs:
            image = obs["frontview_image"]
        elif "agentview_image" in obs:
            image = obs["agentview_image"]
        else:
            # Fallback: try to find any camera observation
            camera_keys = [k for k in obs.keys() if "image" in k]
            if camera_keys:
                image = obs[camera_keys[0]]
            else:
                # Create dummy image if no camera found
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                print("⚠️  Warning: No camera observation found, using dummy image")
        
        # Ensure image is uint8 RGB format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Define the instruction for pick-and-place task
        instruction = "pick up the cube and place it in the target location"
        
        return image, instruction
    
    def _print_action_breakdown(self, action):
        """Print human-readable breakdown of VLA action"""
        print(f"📊 VLA Action Breakdown (7-DoF):")
        print(f"  X position: {action[0]:.4f}")
        print(f"  Y position: {action[1]:.4f}")
        print(f"  Z position: {action[2]:.4f}")
        print(f"  Roll:       {action[3]:.4f}")
        print(f"  Pitch:      {action[4]:.4f}")
        print(f"  Yaw:        {action[5]:.4f}")
        print(f"  Gripper:    {action[6]:.4f}")
    
    def _convert_vla_to_robosuite(self, vla_action, obs):
        """
        Convert VLA model output to robosuite action format
        
        VLA output: [x, y, z, roll, pitch, yaw, gripper]
        Robosuite expects: 7-DOF joint positions or velocities
        """
        # Get current end-effector info for context
        ee_pos = obs.get("robot0_eef_pos", np.array([0.0, 0.0, 1.0]))
        
        # Method 1: Direct mapping (if VLA outputs joint commands)
        if self._is_joint_space_action(vla_action):
            return vla_action[:7]  # Use first 7 values as joint commands
        
        # Method 2: Cartesian space conversion (if VLA outputs Cartesian deltas)
        else:
            return self._cartesian_to_joint_action(vla_action, obs)
    
    def _is_joint_space_action(self, action):
        """Determine if VLA action is in joint space or Cartesian space"""
        max_action = np.max(np.abs(action[:7]))
        return max_action > 0.1  # Threshold to determine action type
    
    def _cartesian_to_joint_action(self, vla_action, obs):
        """Convert Cartesian VLA output to joint space action (simplified)"""
        # Extract Cartesian deltas from VLA action
        cart_delta = vla_action[:3]  # [dx, dy, dz]
        
        # Simple proportional mapping (replace with proper inverse kinematics)
        joint_action = np.zeros(7)
        
        # Map Cartesian deltas to joint velocities (simplified)
        joint_action[0] = cart_delta[1] * 10.0  # Base joint responds to Y
        joint_action[1] = cart_delta[2] * 8.0   # Shoulder responds to Z  
        joint_action[2] = cart_delta[0] * 8.0   # Elbow responds to X
        joint_action[3] = -cart_delta[2] * 3.0  # Wrist adjustments
        joint_action[4] = cart_delta[0] * 2.0   
        joint_action[5] = cart_delta[1] * 2.0   
        joint_action[6] = 0.0  # End effector rotation
        
        # Scale actions to reasonable range
        joint_action = np.clip(joint_action, -0.5, 0.5)
        
        return joint_action
    
    def reset(self):
        """Reset policy state for new episode"""
        self.step_count = 0

# =====================================================================
# Demo Function
# =====================================================================

def main():
    """
    Demo of the PickPlace environment with OpenVLA control
    
    You can choose which integration method to use:
    - "direct": Load model directly with HuggingFace
    - "api": Use REST API server
    - "custom": Your custom integration
    """
    
    print("🚀 OpenVLA Integration Demo")
    print("=" * 50)
    
    # Choose integration method
    print("Available integration methods:")
    print("1. 'direct' - Direct model loading (requires GPU + model download)")
    print("2. 'api' - REST API client (requires running server)")
    print("3. 'custom' - Custom integration template")
    print()
    
    # For this demo, we'll use custom (placeholder) integration
    # Change this to "direct" or "api" based on your setup
    integration_type = "custom"
    
    print(f"Using integration method: {integration_type}")
    print()
    
    # Create environment
    print("Creating PickPlace environment...")
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=True,
        render_camera="frontview",
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=500,
        use_object_obs=True,
        use_camera_obs=True,
        reward_shaping=True,
    )
    
    # Create OpenVLA policy
    print("🤖 Initializing OpenVLA Policy...")
    policy = OpenVLAPolicy(integration_type=integration_type)
    
    # Reset environment and get observation keys
    obs = env.reset()
    print(f"Observation keys: {list(obs.keys())}")
    
    # Debug action space
    low, high = env.action_spec
    print(f"Action space size: {len(low)}")
    print()
    
    print("🎯 Starting OpenVLA Control Demo...")
    print("The robot will be controlled by OpenVLA model predictions")
    print("Press Ctrl+C to stop")
    print()
    
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
            
            while not done:
                # Get action from OpenVLA
                action = policy.get_action(obs)
                
                # Take step
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Render
                env.render()
                
                # Print progress
                if reward > 0 or step % 50 == 0:
                    print(f"Step {step}: Reward = {reward:.3f}")
                
                # Print end-effector position for debugging
                if step % 100 == 0:
                    ee_pos = obs.get("robot0_eef_pos", [0,0,0])
                    print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                
                # Check if episode is done
                if done:
                    print(f"✅ Episode {episode} finished after {step} steps")
                    print(f"📊 Total reward: {total_reward:.3f}")
                    break
                
                # Stop after reasonable time
                if step > 1000:
                    print(f"⏰ Episode {episode} stopped after {step} steps (timeout)")
                    print(f"📊 Total reward: {total_reward:.3f}")
                    break
            
            # Brief pause between episodes
            import time
            time.sleep(3.0)
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    finally:
        env.close()
        print("🔒 Environment closed.")


if __name__ == "__main__":
    main()
