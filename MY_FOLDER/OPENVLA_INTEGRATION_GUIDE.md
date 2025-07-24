# OpenVLA Integration Guide

This guide shows you how to integrate OpenVLA with your robosuite environment.

## Integration Options

### Option 1: Direct Model Loading (Recommended for GPU setups)

**Requirements:**
```bash
pip install transformers torch torchvision timm tokenizers
pip install flash-attn  # Optional but recommended for speed
```

**Usage:**
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load OpenVLA model
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # Optional
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

# Use with your policy
policy = VLAModelPolicy(vla_model=vla_model, processor=processor)
```

### Option 2: REST API Server-Client (Recommended for remote/distributed setups)

**Setup Server:**
```bash
# Install server dependencies
pip install uvicorn fastapi json-numpy

# Start OpenVLA server
python vla-scripts/deploy.py --openvla_path openvla/openvla-7b --host 0.0.0.0 --port 8000
```

**Setup Client:**
```bash
# Install client dependencies  
pip install requests json-numpy
```

**Usage:**
```python
import requests

# Create API client
api_session = requests.Session()

# Use with your policy
policy = VLAModelPolicy(vla_model=api_session, processor=None)
```

### Option 3: Custom Integration

Replace the placeholder methods in `VLAModelPolicy._get_vla_prediction()` with your custom model loading and inference code.

## Running the Demo

1. **Test with placeholder (current setup):**
   ```bash
   python pick_place_robosuite.py
   ```

2. **Test with complete integration examples:**
   ```bash
   python openvla_integration_complete.py
   ```

## Key Integration Points

### Input Format
OpenVLA expects:
- **image**: RGB numpy array (H, W, 3) of type uint8
- **instruction**: String describing the task (e.g., "pick up the cube and place it in the target location")
- **unnorm_key**: Optional string for action unnormalization (use "bridge_orig" for BridgeData V2)

### Output Format  
OpenVLA returns:
- **action**: 7-element numpy array [x, y, z, roll, pitch, yaw, gripper]

### Action Space Conversion
The framework handles conversion between VLA Cartesian outputs and robosuite joint commands. You may need to tune the conversion parameters in `_cartesian_to_joint_action()` for optimal performance.

## Your Current VLA Output
Based on your previous message, your VLA model outputs:
```
[-0.00848473, 0.01025995, -0.00102072, 0.01135582, 0.00323859, -0.0065139, 0.0000]
```

This appears to be small Cartesian deltas, which the framework will convert to joint commands automatically.

## Next Steps

1. **Choose your integration method** (direct loading vs REST API)
2. **Install the required dependencies** for your chosen method
3. **Replace the placeholder** in `VLAModelPolicy._get_vla_prediction()` with your actual model
4. **Test and tune** the action space conversion if needed

## Troubleshooting

- **GPU Memory**: OpenVLA-7B requires ~14GB GPU memory
- **Model Download**: First run will download ~7GB model files
- **Dependencies**: Make sure all required packages are installed
- **Action Scale**: If robot moves too fast/slow, adjust scaling in `_cartesian_to_joint_action()`

Let me know which integration method you'd like to use and I can help you implement it!
