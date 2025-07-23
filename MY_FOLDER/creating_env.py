# run_empty_arena_with_robot.py
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.empty_arena import EmptyArena
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.objects import BallObject
from mujoco.viewer import launch, launch_passive
import mujoco
from mujoco.viewer import launch
import numpy as np
import os

# 1) Create the world container
world = MujocoWorldBase()

# 2) Merge in the empty arena (just the default ground plane)
arena = EmptyArena()
world.merge(arena)

# 3) Add the Panda robot + its gripper
robot   = Panda()
gripper = gripper_factory("PandaGripper")
robot.add_gripper(gripper)
robot.set_base_xpos([0, 0, 0])   # place robot at the world origin
world.merge(robot)

# 4) Create and position a single sphere above the ground
sphere = BallObject(
    name="sphere",
    size=[0.04],               # 4 cm radius
    rgba=[0, 0.5, 0.5, 1]      # teal color
).get_obj()
sphere.set("pos", "0.5 0 0.04")  # x=0.5 m in front of robot, z=radius
world.worldbody.append(sphere)

# 5) Build the native MuJoCo model & state
model = world.get_model(mode="mujoco")
data  = mujoco.MjData(model)


# Control the robot
# 1) Bring up a non-blocking viewer
viewer = launch_passive(model, data)

# 2) Figure out your action dimension (number of actuators)
act_dim = model.nu   # `nu` is # of control inputs

# 3) Main loop: sample a random vector in [-82,12], send it, step, render
while data.time < 1000:  # run for 10 seconds
    random_action = np.random.uniform(-80.0, 80.0, size=act_dim)
    data.ctrl[:]    = random_action        # send to the robot
    mujoco.mj_step(model, data)            # advance physics one frame
    viewer.sync()                          # update the on-screen window



# Add this to your existing code after creating model and data
def save_frame(model, data, filename, width=640, height=480, camera_id=None):
    """Save a frame from the simulation"""
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # Set camera (optional - if None, uses free camera)
    if camera_id is not None:
        renderer.update_scene(data, camera=camera_id)
    else:
        renderer.update_scene(data)
    
    # Render and get pixels
    pixels = renderer.render()
    
    # Convert to PIL Image and save
    image = Image.fromarray(pixels)
    image.save(filename)
    print(f"Frame saved as {filename}")

# Modified main loop with frame saving
frame_counter = 0
save_every_n_frames = 30  # Save every 30 frames (adjust as needed)

while data.time < 10:  # run for 10 seconds
    random_action = np.random.uniform(-80.0, 80.0, size=act_dim)
    data.ctrl[:] = random_action
    mujoco.mj_step(model, data)
    viewer.sync()
    
    # Save frame occasionally
    if frame_counter % save_every_n_frames == 0:
        filename = f"frame_{frame_counter:06d}.png"
        save_frame(model, data, filename)
    
    frame_counter += 1