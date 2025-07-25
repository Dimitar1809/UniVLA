#!/usr/bin/env python3
"""
test_gpu_setup.py - Test GPU setup for MuJoCo and robosuite
"""

import os
import numpy as np

def test_gpu_availability():
    """Test if GPU is available through different interfaces"""
    print("🔍 Testing GPU Availability")
    print("=" * 50)
    
    # Test 1: PyTorch CUDA
    try:
        import torch
        print(f"✅ PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    # Test 2: OpenGL
    try:
        import OpenGL.GL as gl
        print(f"✅ OpenGL available")
        print(f"   Version: {gl.glGetString(gl.GL_VERSION)}")
        print(f"   Renderer: {gl.glGetString(gl.GL_RENDERER)}")
    except ImportError:
        print("⚠️  OpenGL not available")
    
    # Test 3: MuJoCo rendering modes
    print(f"\n🎮 Testing MuJoCo Rendering Modes")
    
    try:
        import mujoco
        print(f"✅ MuJoCo version: {mujoco.__version__}")
        
        # Test different GL backends
        backends = ['egl', 'osmesa', 'glfw']
        for backend in backends:
            try:
                os.environ['MUJOCO_GL'] = backend
                # Try to create a simple model
                xml = """
                <mujoco>
                    <worldbody>
                        <body name="box" pos="0 0 0">
                            <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
                        </body>
                    </worldbody>
                </mujoco>
                """
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                
                # Try to create renderer
                renderer = mujoco.Renderer(model, height=256, width=256)
                mujoco.mj_forward(model, data)
                pixels = renderer.render()
                
                print(f"✅ {backend.upper()} backend works (rendered {pixels.shape})")
                renderer.close()
                
            except Exception as e:
                print(f"❌ {backend.upper()} backend failed: {e}")
                
    except ImportError:
        print("⚠️  MuJoCo not available")

def test_robosuite_minimal():
    """Test minimal robosuite environment creation"""
    print(f"\n🤖 Testing Minimal Robosuite Environment")
    print("=" * 50)
    
    try:
        import robosuite as suite
        
        # Test environment creation with different rendering modes
        configs = [
            ("GPU EGL", {"has_renderer": False, "has_offscreen_renderer": True, "use_camera_obs": True}),
            ("Software OSMesa", {"has_renderer": False, "has_offscreen_renderer": True, "use_camera_obs": True}),
            ("No Rendering", {"has_renderer": False, "has_offscreen_renderer": False, "use_camera_obs": False}),
        ]
        
        for name, config in configs:
            try:
                print(f"\n   Testing {name}...")
                
                # Set appropriate GL backend
                if "GPU" in name:
                    os.environ['MUJOCO_GL'] = 'egl'
                    os.environ['EGL_DEVICE_ID'] = '0'
                elif "Software" in name:
                    os.environ['MUJOCO_GL'] = 'osmesa'
                
                env = suite.make(
                    env_name="Lift",
                    robots="Panda",
                    **config
                )
                
                # Try a quick reset
                obs = env.reset()
                print(f"   ✅ {name} works - obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
                
                # Check for camera observations
                if config.get("use_camera_obs", False):
                    image_keys = [k for k in obs.keys() if "image" in k] if isinstance(obs, dict) else []
                    if image_keys:
                        img = obs[image_keys[0]]
                        print(f"      📷 Camera image shape: {img.shape}")
                    else:
                        print(f"      ⚠️  No camera images found")
                
                env.close()
                
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
                
    except ImportError:
        print("⚠️  Robosuite not available")

def main():
    """Run all GPU tests"""
    print("🧪 GPU Setup Test for OpenVLA + Robosuite")
    print("=" * 60)
    
    test_gpu_availability()
    test_robosuite_minimal()
    
    print(f"\n✅ GPU test completed!")
    print(f"\n💡 Recommendations:")
    print(f"   - If EGL works, use MUJOCO_GL=egl for best performance")
    print(f"   - If EGL fails, use MUJOCO_GL=osmesa for software rendering")
    print(f"   - WSL users may need to use software rendering")

if __name__ == "__main__":
    main()
