# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    # attn_implementation="flash_attention_2",  # Remove this line
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Load image from saved frames folder instead of camera
image_path = "viewer_frames/frame_000510.png"  # or "viewer_frames/frame_000510.png" if that's your folder name
image: Image.Image = Image.open(image_path)

# Format prompt to ask about picking up the green ball
instruction = "pick up the green ball"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Display the predicted actions
print(f"\n🎯 Predicted Robot Actions:")
print(f"Raw action tensor shape: {action.shape}")
print(f"Action values: {action}")  # Remove .cpu().numpy() since it's already numpy

# If it's a 7-DoF action (typical for robot arms)
if len(action) >= 7:
    print(f"\n📊 Action Breakdown (7-DoF):")
    print(f"  X position: {action[0]:.4f}")  # Remove .item() since it's numpy
    print(f"  Y position: {action[1]:.4f}")
    print(f"  Z position: {action[2]:.4f}")
    print(f"  Roll:       {action[3]:.4f}")
    print(f"  Pitch:      {action[4]:.4f}")
    print(f"  Yaw:        {action[5]:.4f}")
    print(f"  Gripper:    {action[6]:.4f}")