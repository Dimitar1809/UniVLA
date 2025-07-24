# OpenVLA + Robosuite Integration Pipeline

This directory contains a complete pipeline for integrating OpenVLA with robosuite for vision-based robot control.

## 📁 Files Overview

### Main Integration Scripts
- **`vla_integration_example.py`** - Complete OpenVLA + robosuite integration pipeline
- **`test_integration_pipeline.py`** - Testing version with dummy VLA outputs  
- **`setup_vla_integration.py`** - Setup script for dependencies and validation

### Supporting Files
- **`GPU_INTEGRATION_GUIDE.md`** - Comprehensive guide for GPU machine setup
- **`pick_place_vision.py`** - Vision-based pick and place (handcrafted policy)
- **`openvla_integration_complete.py`** - Original integration framework

## 🚀 Quick Start

### Option 1: Test Pipeline (Recommended First)
```bash
# 1. Setup environment
python setup_vla_integration.py

# 2. Test with dummy VLA outputs (no model loading)
python test_integration_pipeline.py
```

### Option 2: Full OpenVLA Integration
```bash
# 1. Setup environment (if not done)
python setup_vla_integration.py

# 2. Run full OpenVLA integration
python vla_integration_example.py
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Feed   │───▶│   OpenVLA Model   │───▶│  Action Output  │
│    (256x256)    │    │  (Vision + Lang)  │    │ [x,y,z,r,p,y,g] │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Robot Control  │◀───│  IK Controller   │◀───│Action Converter │
│   (Joint Pos)   │    │   (Robosuite)    │    │ VLA → IK Input  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 Key Features

### OpenVLAController Class
- **Model Loading**: Automatic OpenVLA model and processor loading
- **Action Conversion**: VLA output → robosuite IK controller input
- **Gripper Control**: Threshold-based gripper command extraction
- **Safety Limits**: Action clipping and scaling for safe operation

### VLAPickPlaceEnvironment Class  
- **IK Integration**: Proper robosuite IK controller setup
- **Camera Obs**: Multi-camera observations for VLA model
- **Action Space**: 6D IK actions + gripper control
- **Visualization**: Real-time rendering and debugging

## 🔧 Technical Details

### VLA Action Format
```python
VLA Output: [x, y, z, roll, pitch, yaw, gripper]
```

### Robosuite IK Input Format
```python
IK Input: [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z]
```

### Action Conversion Process
1. **Extract Deltas**: Position and orientation changes from VLA
2. **Scale Actions**: Apply safety scaling factor
3. **Clip Values**: Ensure actions stay within safe bounds
4. **Send to IK**: Robosuite IK controller handles inverse kinematics

### Gripper Control Logic
```python
if gripper_command > 0.5:  # Open gripper
if gripper_command < -0.5: # Close gripper
else:                      # Maintain current state
```

## 🎮 Controller Options

The pipeline supports multiple robosuite IK controllers:

### IK_POSE (Recommended)
- Direct Cartesian pose control
- 6D action space: [dx, dy, dz, dax, day, daz]
- Built-in inverse kinematics solver

### OSC_POSE
- Operational Space Controller
- More advanced dynamics handling
- Better for precise manipulation

### WholeBodyIK
- Full-body inverse kinematics
- Handles complex multi-arm scenarios

## 📊 Performance Parameters

### Default Settings
- **Control Frequency**: 20 Hz
- **Action Scaling**: 1.0 (adjust for your VLA model)
- **Max Action Magnitude**: 0.05 (safety limit)
- **Episode Length**: 500-1000 steps
- **Camera Resolution**: 256x256 RGB

### Tunable Parameters
```python
# In OpenVLAController.__init__()
self.action_scaling = 1.0          # Adjust VLA action magnitude
self.max_action_magnitude = 0.05   # Safety clipping limit

# In VLAPickPlaceEnvironment.__init__()
control_freq = 20                  # Control frequency (Hz)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. GPU Memory Error
```
CUDA out of memory
```
**Solution**: Use smaller model or reduce batch size:
```python
torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float32
low_cpu_mem_usage=True,      # Enable CPU memory optimization
```

#### 2. Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution**: Run setup script:
```bash
python setup_vla_integration.py
```

#### 3. Action Space Mismatch
```
Action dimension mismatch
```
**Solution**: Check controller configuration:
```python
# Ensure controller matches expected action space
controller_config = load_composite_controller_config(controller="IK_POSE")
```

#### 4. VLA Model Loading Fails
```
Failed to load OpenVLA model
```
**Solution**: Check internet connection and HuggingFace access:
```bash
huggingface-cli login  # If using private models
```

### Debug Tips

#### Enable Verbose Logging
```python
# In main()
print("🎯 Step X - VLA Action Analysis:")
print(f"   VLA Raw: {vla_action}")
print(f"   IK Action: {ik_action}")
```

#### Test Components Separately
1. **Test Environment Only**: Run `test_integration_pipeline.py`
2. **Test VLA Only**: Load model in isolation
3. **Test Controllers**: Try different controller types

#### Monitor Resource Usage
```bash
# GPU memory
nvidia-smi

# CPU/RAM usage  
htop
```

## 🔄 Integration Flow

### Episode Loop
1. **Environment Reset**: Initialize robosuite environment
2. **Observation**: Get camera image + robot state
3. **VLA Inference**: Process image + instruction → action
4. **Action Conversion**: VLA output → IK controller input
5. **Environment Step**: Execute action, get reward
6. **Repeat**: Continue until episode termination

### Key Code Points
```python
# VLA inference
vla_action = vla_controller.get_vla_action(image)

# Action conversion
ik_action = vla_controller.convert_vla_to_robosuite_action(vla_action)
gripper_action = vla_controller.get_gripper_action(vla_action)

# Environment step
obs, reward, done, info = env.step(ik_action, gripper_action)
```

## 📝 Next Steps

1. **Test on Your Hardware**: Run test pipeline first
2. **Tune Parameters**: Adjust scaling factors for your VLA model
3. **Add Task Variations**: Extend to different manipulation tasks
4. **Optimize Performance**: Profile and optimize inference speed
5. **Add Safety Features**: Implement collision avoidance

## 💡 Tips for Success

- **Start with Testing**: Use `test_integration_pipeline.py` first
- **Check GPU Memory**: OpenVLA needs 6-8GB+ GPU memory
- **Tune Action Scaling**: Different VLA models may need different scaling
- **Monitor Safety**: Always use action clipping and magnitude limits
- **Validate Setup**: Run `setup_vla_integration.py` before starting

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Validate your setup with the test scripts
3. Review the GPU_INTEGRATION_GUIDE.md for detailed setup
4. Ensure all dependencies are properly installed
