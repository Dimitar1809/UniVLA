#!/usr/bin/env python3
"""
Test script that exactly matches OpenVLA LIBERO evaluation setup
"""
import numpy as np
import robosuite as suite
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

def create_libero_env():
    """Create environment exactly like OpenVLA LIBERO evaluation"""
    
    # Load OSC controller config (like OpenVLA)
    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    
    env = suite.make(
        env_name="Lift",
        robots="Panda", 
        controller_configs=controller_configs,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],  # Like LIBERO
        camera_heights=84,  # LIBERO uses 84x84 images!
        camera_widths=84,
        control_freq=20,
        horizon=600,  # LIBERO horizon
    )
    
    return env

def process_image_libero_style(image):
    """Process image exactly like LIBERO"""
    # LIBERO uses 84x84 images, not 256x256!
    if image.shape != (84, 84, 3):
        # Resize to 84x84 if needed
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        pil_img = pil_img.resize((84, 84))
        image = np.array(pil_img)
    
    return PILImage.fromarray(image)

def test_libero_exact():
    """Test with exact LIBERO setup"""
    print("🧪 Testing with EXACT OpenVLA LIBERO setup")
    
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b-finetuned-libero-goal", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Create LIBERO environment
    env = create_libero_env()
    obs = env.reset()
    
    robot = env.robots[0]
    arm = robot.arms[0]
    
    for step in range(200):
        # Get image (LIBERO style)
        if "agentview_image" in obs:
            image = obs["agentview_image"]
        else:
            image = obs["frontview_image"]
        
        # Process image LIBERO style
        pil_image = process_image_libero_style(image)
        
        # Get VLA action
        prompt = "pick up the cube"
        inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            action = model.predict_action(**inputs, unnorm_key="libero_goal", do_sample=False)
        
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        # Use VLA action directly (minimal scaling like LIBERO)
        position_action = action[:3] * 0.1  # Very light scaling
        rotation_action = action[3:6] * 0.1
        gripper_action = 0.0 if action[6] > 0.5 else 0.04
        
        # Create action
        ik_action = np.concatenate([position_action, rotation_action])
        full_action = robot.create_action_vector({
            arm: ik_action,
            f"{arm}_gripper": np.array([gripper_action])
        })
        
        obs, reward, done, info = env.step(full_action)
        
        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, action_mag={np.linalg.norm(action[:6]):.3f}")
        
        if done:
            break
    
    env.close()

if __name__ == "__main__":
    test_libero_exact()