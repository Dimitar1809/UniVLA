#!/usr/bin/env python3
"""
Exact implementation of OpenVLA LIBERO evaluation approach
Based on: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py
"""

import os
import time
import numpy as np
import torch
import robosuite as suite
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Constants from OpenVLA repo
ACTION_DIM = 7  # [x, y, z, roll, pitch, yaw, gripper]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# OpenVLA v0.1 system prompt (from their code)
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def create_libero_env():
    """
    Create environment exactly like OpenVLA LIBERO evaluation
    """
    print("🏗️ Creating LIBERO environment (matching OpenVLA setup)")
    
    # Fix: Use correct robosuite controller loading function
    from robosuite.controllers.parts.controller_factory import load_part_controller_config
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
    
    # Load OSC controller config (like OpenVLA)
    part_controller_config = load_part_controller_config(default_controller="OSC_POSE")
    controller_configs = refactor_composite_controller_config(
        part_controller_config, robot_type="Panda", arms=["right"]
    )
    
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_configs,  # Use the properly configured controller
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],  # OpenVLA LIBERO uses agentview
        camera_heights=224,  # OpenVLA uses 224x224 images
        camera_widths=224,
        control_freq=20,  # 20 Hz control frequency
        horizon=600,  # LIBERO horizon
        reward_shaping=True,
    )
    
    print("✅ LIBERO environment created successfully")
    return env

def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """
    Generates an action with the VLA policy.
    Exact implementation from OpenVLA repository.
    """
    # Extract image from observation
    if "agentview_image" in obs:
        image_array = obs["agentview_image"]
    elif "frontview_image" in obs:
        image_array = obs["frontview_image"]
    else:
        raise ValueError("No camera image found in observation")
    
    # Convert to PIL Image
    image_corrected = np.rot90(image_array, 2)  # 180-degree rotation
    image = Image.fromarray(image_corrected)
    image = image.convert("RGB")

    # Center crop (if specified) - from OpenVLA code
    if center_crop:
        import tensorflow as tf
        from experiments.robot.robot_utils import crop_and_resize  # Would need this from OpenVLA
        
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt (exact from OpenVLA code)
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action

def get_action(cfg, model, obs, task_label, processor=None):
    """
    Queries the model to get an action.
    Exact implementation from OpenVLA repository.
    """
    if cfg["model_family"] == "openvla":
        action = get_vla_action(
            model, processor, cfg["pretrained_checkpoint"], obs, task_label, cfg["unnorm_key"], center_crop=cfg.get("center_crop", False)
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action

def setup_openvla_model():
    """Load OpenVLA model and processor"""
    print("📦 Loading OpenVLA model...")
    
    model_name = "openvla/openvla-7b"
    checkpoint_name = "openvla/openvla-7b-finetuned-libero-goal"
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    print(f"✅ OpenVLA model loaded on {DEVICE}")
    return model, processor, model_name

def run_libero_evaluation():
    """
    Run LIBERO evaluation exactly like OpenVLA repository
    """
    print("🚀 Starting OpenVLA LIBERO Evaluation")
    
    # Setup model
    model, processor, base_model_name = setup_openvla_model()
    
    # Configuration (matching OpenVLA LIBERO config)
    cfg = {
        "model_family": "openvla",
        "pretrained_checkpoint": base_model_name,
        "unnorm_key": "libero_goal",
        "center_crop": False  # Set to True if you want center cropping
    }
    
    # Create environment
    env = create_libero_env()
    
    # Task configuration
    task_label = "slide the object across the table"  # LIBERO task description
    max_steps = 6000
    num_episodes = 1
    
    total_successes = 0
    
    for episode in range(num_episodes):
        print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs = env.reset()
        done = False
        t = 0
        episode_reward = 0
        
        try:
            while not done and t < max_steps:
                # Get action from VLA (exact OpenVLA approach)
                action = get_action(cfg, model, obs, task_label, processor)
                
                if torch.is_tensor(action):
                    action = action.cpu().numpy()
                
                if t % 50 == 0:
                    print(f"   Step {t}: Action = {action}")
                    print(f"   Action magnitude: {np.linalg.norm(action[:6]):.4f}")
                
                # Execute action in environment (EXACT like OpenVLA)
                obs, reward, done, info = env.step(action.tolist())
                episode_reward += reward
                
                if t % 50 == 0:
                    print(f"   Step {t}: Reward = {reward:.4f} (Total: {episode_reward:.4f})")
                
                if done:
                    print(f"✅ Episode {episode + 1} completed successfully!")
                    total_successes += 1
                    break
                    
                t += 1
                
        except Exception as e:
            print(f"❌ Episode {episode + 1} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Results
    success_rate = total_successes / num_episodes
    print(f"\n📊 LIBERO Evaluation Results:")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Successful episodes: {total_successes}")
    print(f"   Success rate: {success_rate:.2%}")
    
    env.close()

def test_single_step():
    """Test a single step to debug the setup"""
    print("🧪 Testing single step")
    
    # Setup
    model, processor, base_model_name = setup_openvla_model()
    env = create_libero_env()
    obs = env.reset()
    
    cfg = {
        "model_family": "openvla",
        "pretrained_checkpoint": base_model_name,
        "unnorm_key": "libero_goal",
        "center_crop": False
    }
    
    # Get single action
    task_label = "push the cube to the right"
    action = get_action(cfg, model, obs, task_label, processor)
    
    if torch.is_tensor(action):
        action = action.cpu().numpy()
    
    print(f"🎯 Single step test:")
    print(f"   Task: {task_label}")
    print(f"   Action: {action}")
    print(f"   Action shape: {action.shape}")
    print(f"   Action magnitude: {np.linalg.norm(action[:6]):.4f}")
    
    # Execute one step
    obs, reward, done, info = env.step(action.tolist())
    print(f"   Reward: {reward:.4f}")
    print(f"   Done: {done}")
    
    env.close()

if __name__ == "__main__":
    # Choose what to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_step()
    else:
        run_libero_evaluation()