#!/bin/bash
"""
setup_vla_integration.py - Setup script for VLA integration pipeline

This script sets up the environment and dependencies for the VLA integration.
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU found: {gpu_name}")
            print(f"   CUDA devices: {gpu_count}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU found - will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_vla_integration():
    """Main setup function"""
    print("🚀 OpenVLA + Robosuite Integration Setup")
    print("=" * 60)
    
    # Check current directory
    if not os.path.exists("vla_integration_example.py"):
        print("❌ Please run this script from the GateInternship directory")
        return False
    
    print("✅ Found VLA integration files")
    
    # Check Python version
    python_version = sys.version_info
    print(f"\n🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    
    # Install basic dependencies
    dependencies = [
        ("torch torchvision torchaudio", "Installing PyTorch"),
        ("transformers", "Installing Transformers"),
        ("Pillow", "Installing PIL (Pillow)"),
        ("robosuite", "Installing robosuite"),
        ("numpy", "Installing NumPy"),
        ("opencv-python", "Installing OpenCV"),
    ]
    
    for package, description in dependencies:
        if not run_command(f"pip install {package}", description):
            print(f"❌ Failed to install {package}")
            return False
    
    # Check robosuite installation
    print("\n🤖 Checking robosuite installation...")
    try:
        import robosuite
        print(f"✅ Robosuite version: {robosuite.__version__}")
    except ImportError:
        print("❌ Robosuite not properly installed")
        return False
    
    # Check GPU
    check_gpu()
    
    # Test basic imports
    print("\n🧪 Testing imports...")
    test_imports = [
        ("import torch", "PyTorch"),
        ("import transformers", "Transformers"),
        ("from PIL import Image", "PIL"),
        ("import robosuite", "Robosuite"),
        ("import numpy as np", "NumPy"),
        ("import cv2", "OpenCV"),
    ]
    
    for import_cmd, name in test_imports:
        try:
            exec(import_cmd)
            print(f"✅ {name} import successful")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            return False
    
    print("\n🎯 Setup Validation")
    print("-" * 30)
    
    # Check if we can load a controller config
    try:
        from robosuite import load_composite_controller_config
        config = load_composite_controller_config("IK_POSE")
        print("✅ Robosuite IK controller config loaded")
    except Exception as e:
        print(f"❌ Failed to load controller config: {e}")
        return False
    
    # Test environment creation (without OpenVLA model)
    print("🧪 Testing environment creation...")
    try:
        import robosuite as suite
        from robosuite.wrappers.gym_wrapper import GymWrapper
        
        raw_env = suite.make(
            env_name="PickPlace",
            robots="Panda", 
            has_renderer=False,  # No rendering for test
            horizon=10,
        )
        env = GymWrapper(raw_env)
        obs = env.reset()
        env.close()
        print("✅ Environment creation test passed")
    except Exception as e:
        print(f"❌ Environment creation test failed: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. For testing without OpenVLA model:")
    print("   python test_integration_pipeline.py")
    print("\n2. For full OpenVLA integration:")
    print("   python vla_integration_example.py")
    print("\n3. Make sure you have sufficient GPU memory (8GB+ recommended)")
    
    return True

if __name__ == "__main__":
    success = setup_vla_integration()
    sys.exit(0 if success else 1)
