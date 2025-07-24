# OpenVLA + Robosuite Integration Guide for A5000 GPU Machine

## Overview

This guide provides step-by-step instructions for integrating OpenVLA with robosuite using proper inverse kinematics controllers on your A5000 GPU machine. Based on the robosuite documentation, we'll use the built-in IK controllers instead of manual action space conversion.

## Step 1: Environment Setup

### 1.1 Install Core Dependencies
```bash
# Create conda environment (recommended)
conda create -n openvla python=3.10
conda activate openvla

# Install PyTorch with CUDA support for A5000
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install OpenVLA dependencies (pinned versions from robosuite docs)
pip install transformers==4.40.1
pip install timm==0.9.10
pip install tokenizers==0.19.1

# Install other required packages
pip install pillow numpy

# Install flash-attention for faster inference (optional but recommended)
pip install packaging ninja
pip install flash-attn==2.5.5 --no-build-isolation

# Install robosuite
pip install robosuite
```

### 1.2 Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Step 2: Understanding Robosuite IK Controllers

### 2.1 Controller Options for OpenVLA Integration

Based on robosuite documentation, you have these controller options:

1. **IK_POSE** (Recommended for OpenVLA)
   - Action Dimensions: 6 (position + orientation deltas)
   - Automatically handles inverse kinematics
   - Input: `[dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z]`
   - Perfect match for OpenVLA's `[x, y, z, roll, pitch, yaw, gripper]` output

2. **OSC_POSE** (Operational Space Control)
   - Action Dimensions: 6 
   - More advanced dynamics-aware control
   - Input: `[dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z]`

3. **WholeBodyIK** (Composite Controller)
   - Takes end-effector targets, converts to joint angles automatically
   - Ideal for complex robots with multiple parts

### 2.2 Key Differences Between Controllers

- **IK controllers**: Rotation axes relative to end-effector frame
- **OSC controllers**: Rotation axes relative to global world frame
- **Both interpret actions as DELTA values by default** (perfect for OpenVLA)

## Step 3: OpenVLA Model Loading

### 3.1 Load OpenVLA Model
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

# Model loading
model_name = "openvla/openvla-7b"
device = torch.device("cuda:0")

print("Loading OpenVLA processor...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print("Loading OpenVLA model...")
vla_model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # Use flash attention if available
    torch_dtype=torch.bfloat16,               # Use bfloat16 for efficiency
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device)

print(f"✅ OpenVLA model loaded on {device}")
```

### 3.2 Test Model Inference
```python
from PIL import Image
import numpy as np

# Test with dummy image
test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
instruction = "pick up the cube"

# Format prompt for OpenVLA
prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

# Run inference
inputs = processor(prompt, test_image).to(device, dtype=torch.bfloat16)
with torch.no_grad():
    action = vla_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Test action output: {action}")
```

## Step 4: Robosuite Environment Configuration

### 4.1 Configure IK Controller
```python
import robosuite as suite
from robosuite import load_composite_controller_config

# Option 1: Use IK_POSE controller (recommended)
config = load_composite_controller_config(controller="IK_POSE")

# Option 2: Use OSC_POSE controller (alternative)
# config = load_composite_controller_config(controller="OSC_POSE")

# Option 3: Use WholeBodyIK controller (for complex scenarios)
# config = load_composite_controller_config(controller="WholeBodyIK")
```

### 4.2 Create Environment with IK Control
```python
env = suite.make(
    env_name="PickPlace",           # or "Lift", "Stack", etc.
    robots="Panda",                 # Robot type
    controller_configs=config,      # Use IK controller
    has_renderer=True,              # Enable visualization
    render_camera="frontview",      # Camera for rendering
    has_offscreen_renderer=False,   # No offscreen rendering needed
    control_freq=20,                # 20 Hz control frequency
    horizon=500,                    # Episode length
    use_object_obs=True,            # Object state observations
    use_camera_obs=True,            # Camera observations for VLA
    reward_shaping=True,            # Shaped rewards
)

# Print action space information
env.robots[0].print_action_info()
```

## Step 5: Action Space Conversion Strategy

### 5.1 OpenVLA Output Analysis
Your VLA outputs: `[-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000]`

This maps to:
- `[0:3]`: Position deltas `[dx, dy, dz]`
- `[3:6]`: Orientation deltas `[droll, dpitch, dyaw]` 
- `[6]`: Gripper command (not used in current action space)

### 5.2 Conversion Function for IK Controller
```python
def convert_openvla_to_ik_action(vla_output):
    """
    Convert OpenVLA output to robosuite IK controller input
    
    VLA Output: [x, y, z, roll, pitch, yaw, gripper]
    IK Input:   [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z]
    """
    # Extract components
    position_delta = vla_output[:3]      # [dx, dy, dz]
    orientation_delta = vla_output[3:6]  # [droll, dpitch, dyaw]
    
    # For IK controller, we need to convert euler angles to axis-angle
    # Simple approach: use roll, pitch, yaw directly as axis-angle components
    # (This is an approximation - for precise control, convert via rotation matrices)
    
    ik_action = np.concatenate([position_delta, orientation_delta])
    
    # Scale actions to appropriate range for robosuite
    # IK controllers typically expect small delta values
    scaling_factor = 1.0  # Adjust based on robot response
    ik_action = ik_action * scaling_factor
    
    # Clip to safe ranges (adjust limits based on your setup)
    ik_action = np.clip(ik_action, -0.05, 0.05)
    
    return ik_action
```

### 5.3 Alternative: Use OSC Controller Conversion
```python
def convert_openvla_to_osc_action(vla_output):
    """
    Convert OpenVLA output to robosuite OSC controller input
    
    OSC takes same 6D input but interprets rotations relative to world frame
    """
    position_delta = vla_output[:3]
    orientation_delta = vla_output[3:6]
    
    osc_action = np.concatenate([position_delta, orientation_delta])
    
    # OSC may handle larger action magnitudes
    scaling_factor = 2.0  # Adjust based on testing
    osc_action = osc_action * scaling_factor
    
    return np.clip(osc_action, -0.1, 0.1)
```

## Step 6: Integration Implementation

### 6.1 Complete VLA Policy Class
```python
class OpenVLAPolicy:
    def __init__(self, vla_model, processor, controller_type="IK_POSE"):
        self.vla_model = vla_model
        self.processor = processor
        self.controller_type = controller_type
        self.step_count = 0
        
    def get_action(self, obs):
        """Get action from OpenVLA for robosuite"""
        # Extract image from observation
        image = self._extract_image(obs)
        
        # Get VLA prediction
        vla_action = self._get_vla_prediction(image)
        
        # Convert to robosuite action space
        if self.controller_type == "IK_POSE":
            robosuite_action = convert_openvla_to_ik_action(vla_action)
        elif self.controller_type == "OSC_POSE":
            robosuite_action = convert_openvla_to_osc_action(vla_action)
        else:
            raise ValueError(f"Unsupported controller type: {self.controller_type}")
        
        self.step_count += 1
        return robosuite_action
    
    def _extract_image(self, obs):
        """Extract RGB image from robosuite observation"""
        # Try different camera keys
        for key in ["frontview_image", "agentview_image"]:
            if key in obs:
                image = obs[key]
                break
        else:
            # Find any image key
            image_keys = [k for k in obs.keys() if "image" in k]
            if image_keys:
                image = obs[image_keys[0]]
            else:
                raise ValueError("No camera observation found")
        
        # Ensure proper format for OpenVLA
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
    
    def _get_vla_prediction(self, image):
        """Get prediction from OpenVLA model"""
        instruction = "pick up the cube and place it in the target location"
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Process inputs
        inputs = self.processor(prompt, pil_image).to(
            self.vla_model.device, dtype=torch.bfloat16
        )
        
        # Get action prediction
        with torch.no_grad():
            action = self.vla_model.predict_action(
                **inputs, 
                unnorm_key="bridge_orig", 
                do_sample=False
            )
        
        # Convert to numpy if needed
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        return action
```

## Step 7: Complete Integration Script Structure

```python
# Main integration script structure
def main():
    # 1. Load OpenVLA model
    processor, vla_model = load_openvla_model()
    
    # 2. Configure robosuite with IK controller
    config = load_composite_controller_config(controller="IK_POSE")
    env = suite.make("PickPlace", robots="Panda", controller_configs=config, ...)
    
    # 3. Create VLA policy
    policy = OpenVLAPolicy(vla_model, processor, controller_type="IK_POSE")
    
    # 4. Run control loop
    obs = env.reset()
    for step in range(500):
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            break
```

## Step 8: Testing and Tuning

### 8.1 Initial Testing
1. Start with small scaling factors in conversion functions
2. Monitor robot behavior for smooth motion
3. Check for any joint limit violations
4. Observe end-effector tracking accuracy

### 8.2 Parameter Tuning
- **Action scaling**: Adjust scaling factors in conversion functions
- **Controller gains**: Modify IK controller parameters if needed
- **Action limits**: Adjust clipping ranges based on task requirements
- **Control frequency**: Test different control frequencies (10-30 Hz)

### 8.3 Debugging Tips
- Print VLA actions and converted actions for comparison
- Use `env.robots[0].print_action_info()` to understand action space
- Monitor end-effector position vs. commanded position
- Check for action saturation (hitting limits frequently)

## Step 9: Advanced Configuration

### 9.1 Custom Controller Configuration
If default controllers need modification, create custom JSON config:

```json
{
  "type": "IK_POSE",
  "interpolation": "linear",
  "ramp_ratio": 0.2,
  "ik_pos_limit": 0.1,
  "ik_ori_limit": 0.2,
  "control_delta": true,
  "kp": 150,
  "damping_ratio": 1.0
}
```

### 9.2 Performance Optimization
- Use `torch.no_grad()` for inference
- Consider model quantization for faster inference
- Batch multiple VLA predictions if needed
- Use asynchronous model loading

## Expected Results

With proper IK controller setup:
- **Smoother robot motion** compared to manual joint control
- **Better end-effector tracking** of VLA commands
- **Automatic handling of joint limits** and singularities
- **More stable control** during complex manipulations

The IK controller automatically handles the complex mathematics of converting your OpenVLA's Cartesian commands into appropriate joint motions, eliminating the need for manual action space conversion functions.
