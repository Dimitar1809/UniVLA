#!/usr/bin/env python3
"""
image_utils.py - Image handling and visualization utilities for VLA-Robosuite integration

This module handles all camera observation extraction, image processing, debugging,
and visualization functions.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def save_and_display_image(image, step, camera_name="unknown", save_dir="debug_images"):
    """
    Save and optionally display the camera image for debugging
    
    Args:
        image: numpy array of image (H, W, 3)
        step: current step number
        camera_name: name of the camera
        save_dir: directory to save images
    """
    if image is None:
        return
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save image every 20 steps to avoid too many files
    if step % 20 == 0:
        filename = f"{save_dir}/step_{step:04d}_{camera_name}.png"
        
        # Convert to RGB if needed and save
        if len(image.shape) == 3:
            # Ensure proper format for saving
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    save_image = (image * 255).astype(np.uint8)
                else:
                    save_image = image.astype(np.uint8)
            else:
                save_image = image
            
            cv2.imwrite(filename, cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved image: {filename} (shape: {image.shape})")
            
            # Print image statistics
            print(f"   Image stats - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}")
            print(f"   Dtype: {image.dtype}, Shape: {image.shape}")


def extract_image_from_obs(obs, step=0, debug=True):
    """
    Extract camera image from robosuite observation (raw environment format)
    Enhanced with image visualization and debugging
    
    Args:
        obs: Observation from raw robosuite environment (should be dict)
        step: Current step number (for debugging)
        debug: Whether to print debug information
        
    Returns:
        numpy array: RGB image (H, W, 3) or None if no image available
    """
    if debug and step % 10 == 0:
        print(f"🔍 Step {step} - Image extraction debug:")
        print(f"   Obs type: {type(obs)}")
    
    # For raw robosuite environment, observation should be a dictionary
    if not isinstance(obs, dict):
        if debug and step % 10 == 0:
            print(f"   Expected dict observation but got: {type(obs)}")
            print("   This suggests the environment might be wrapped with GymWrapper")
        return None
    
    # Look for camera observations
    obs_keys = list(obs.keys())
    if debug and step % 10 == 0:
        print(f"   Available keys: {obs_keys[:10]}...")  # Show first 10 keys
    
    # Find image keys (different possible naming conventions)
    image_keys = []
    for key in obs_keys:
        if any(img_keyword in key.lower() for img_keyword in ['image', 'camera', 'rgb']):
            image_keys.append(key)
    
    if debug and step % 10 == 0:
        print(f"   Found image keys: {image_keys}")
    
    # Try to extract image in order of preference
    preferred_cameras = ['frontview_image', 'agentview_image', 'robot0_eye_in_hand_image']
    
    for camera in preferred_cameras:
        if camera in obs:
            image = obs[camera]
            if image is not None and hasattr(image, 'shape'):
                if debug and step % 10 == 0:
                    print(f"   Using {camera}: {type(image)} {image.shape}")
                
                # Save and analyze the image
                save_and_display_image(image, step, camera.replace('_image', ''))
                
                return image
    
    # If preferred cameras not found, try any available image key
    for key in image_keys:
        image = obs[key]
        if image is not None and hasattr(image, 'shape'):
            if debug and step % 10 == 0:
                print(f"   Using {key}: {type(image)} {image.shape}")
            
            # Save and analyze the image
            save_and_display_image(image, step, key.replace('_image', ''))
            
            return image
    
    # No valid image found
    if debug and step % 10 == 0:
        print("   No valid camera image found")
    return None


def analyze_all_camera_views(obs, step=0, debug=True):
    """
    Analyze and save all available camera views for debugging
    
    Args:
        obs: Observation dictionary
        step: Current step number
        debug: Whether to print debug information
    """
    if not isinstance(obs, dict):
        return
    
    # Save all camera views every 50 steps for comprehensive analysis
    if debug and step % 50 == 0:
        print(f"\n📸 Step {step} - Analyzing ALL camera views:")
        
        image_keys = [k for k in obs.keys() if 'image' in k.lower()]
        
        for key in image_keys:
            image = obs[key]
            if image is not None and hasattr(image, 'shape'):
                print(f"   {key}: {image.shape}, dtype: {image.dtype}")
                print(f"      Range: [{image.min():.3f}, {image.max():.3f}], Mean: {image.mean():.3f}")
                
                # Save each camera view
                save_and_display_image(image, step, key.replace('_image', ''))
                
                # Check if image looks reasonable
                if image.max() <= 1.0:
                    print(f"      ⚠️  Image values in [0,1] range - might need scaling")
                if image.mean() < 0.1:
                    print(f"      ⚠️  Very dark image - check lighting")
                if image.mean() > 0.9:
                    print(f"      ⚠️  Very bright image - might be overexposed")


def save_vla_input_image(image, step, save_dir="debug_images/vla_input"):
    """
    Save the processed image that goes to VLA model
    
    Args:
        image: PIL Image or numpy array
        step: Current step number
        save_dir: Directory to save VLA input images
    """
    if step % 20 == 0:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/vla_input_step_{step:04d}.png"
        
        if isinstance(image, Image.Image):
            image.save(filename)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    save_image = (image * 255).astype(np.uint8)
                else:
                    save_image = image.astype(np.uint8)
            else:
                save_image = image
            cv2.imwrite(filename, cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
        
        print(f"💾 Saved VLA input image: {filename}")


def create_camera_summary_plot(save_dir="debug_images"):
    """
    Create a summary plot showing all camera views from the latest step
    
    Args:
        save_dir: Directory where debug images are saved
    """
    try:
        import glob
        
        # Find latest images
        image_files = glob.glob(f"{save_dir}/step_*.png")
        if not image_files:
            print("No debug images found to create summary")
            return
        
        # Get the latest step
        latest_step = max([int(f.split('_')[1]) for f in image_files])
        latest_images = [f for f in image_files if f"step_{latest_step:04d}" in f]
        
        if len(latest_images) == 0:
            return
        
        # Create subplot
        fig, axes = plt.subplots(1, len(latest_images), figsize=(15, 5))
        if len(latest_images) == 1:
            axes = [axes]
        
        for i, img_file in enumerate(latest_images):
            img = cv2.imread(img_file)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            camera_name = os.path.basename(img_file).split('_')[-1].replace('.png', '')
            axes[i].set_title(f'{camera_name} (Step {latest_step})')
            axes[i].axis('off')
        
        plt.tight_layout()
        summary_file = f"{save_dir}/camera_summary_step_{latest_step:04d}.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Created camera summary: {summary_file}")
        
    except Exception as e:
        print(f"❌ Failed to create camera summary: {e}")


def process_image_for_vla(image, step_count=0):
    """
    Process image for VLA model input with debugging and visualization
    Enhanced to flip images 180 degrees to correct robosuite orientation
    
    Args:
        image: numpy array or PIL Image
        step_count: Current step count
        
    Returns:
        PIL.Image: Processed image ready for VLA model
    """
    if image is None:
        return None
    
    try:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Print original image info
            print(f"🖼️  Original image: {image.shape}, dtype: {image.dtype}")
            print(f"   Range: [{image.min()}, {image.max()}], Mean: {image.mean():.3f}")
            
            # Ensure proper format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                    print("   Converted from [0,1] to [0,255] range")
                else:
                    image = image.astype(np.uint8)
                    print("   Converted to uint8")
            
            # Save original image for comparison (every 20 steps)
            if step_count % 20 == 0:
                save_dir_orig = "debug_images/original_vs_corrected"
                os.makedirs(save_dir_orig, exist_ok=True)
                filename_orig = f"{save_dir_orig}/step_{step_count:04d}_original.png"
                cv2.imwrite(filename_orig, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print(f"💾 Saved ORIGINAL image: {filename_orig}")
            
            # Flip image 180 degrees to correct robosuite orientation
            image_corrected = np.rot90(image, 2)  # Rotate 180 degrees
            print("   Applied 180-degree rotation to correct orientation")
            
            # Save corrected image for comparison (every 20 steps)
            if step_count % 20 == 0:
                filename_corrected = f"{save_dir_orig}/step_{step_count:04d}_corrected_for_vla.png"
                cv2.imwrite(filename_corrected, cv2.cvtColor(image_corrected, cv2.COLOR_RGB2BGR))
                print(f"💾 Saved CORRECTED image that goes to VLA: {filename_corrected}")
            
            pil_image = Image.fromarray(image_corrected)
            
            # Double-check the PIL image we're sending to VLA
            print(f"   🎯 VLA will receive PIL image: size={pil_image.size}, mode={pil_image.mode}")
            
            # Save the exact PIL image that goes to VLA
            save_vla_input_image(pil_image, step_count)
        else:
            pil_image = image
        
        print("Before processing inputs, checking image type...")
        print(f"   Image type: {type(pil_image)}")
        print(f"   PIL Image size: {pil_image.size}")
        print(f"   PIL Image mode: {pil_image.mode}")
        
        return pil_image
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None


def analyze_vla_action(action):
    """
    Analyze and print VLA action statistics
    
    Args:
        action: numpy array of VLA action
    """
    print(f"🎯 VLA Action: {action}")
    print(f"   Position magnitude: {np.linalg.norm(action[:3]):.6f}")
    print(f"   Rotation magnitude: {np.linalg.norm(action[3:6]):.6f}")
    if len(action) > 6:
        print(f"   Gripper: {action[6]:.6f}")


def check_robot_state(obs, step):
    """
    Check and print robot state information
    
    Args:
        obs: Observation dictionary
        step: Current step number
    """
    if step % 50 == 0 and isinstance(obs, dict):
        if "robot0_eef_pos" in obs:
            ee_pos = obs["robot0_eef_pos"]
            print(f"🔍 Step {step} Debug:")
            print(f"   EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            
            # Check if cube is present and where it is
            cube_keys = [k for k in obs.keys() if 'cube' in k.lower() and 'pos' in k.lower()]
            for cube_key in cube_keys:
                cube_pos = obs[cube_key]
                distance_to_cube = np.linalg.norm(ee_pos - cube_pos)
                print(f"   {cube_key}: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
                print(f"   Distance to cube: {distance_to_cube:.3f}")


def clear_debug_images(save_dir="debug_images"):
    """
    Clear all debug images from previous runs
    
    Args:
        save_dir: Directory containing debug images
    """
    try:
        import shutil
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print(f"🧹 Cleared debug images from {save_dir}")
    except Exception as e:
        print(f"⚠️  Could not clear debug images: {e}")


def setup_debug_directories():
    """
    Set up debug directories for image saving
    """
    directories = [
        "debug_images",
        "debug_images/vla_input",
        "debug_images/camera_views",
        "debug_images/summaries"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 Debug directories set up successfully")