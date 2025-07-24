# run_empty_arena_viewgrab.py  –  save *exactly* what the GUI shows
import os, xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from robosuite.models import MujocoWorldBase, arenas, robots, grippers, objects

# -------------------------------------------------------------------- scene
world = MujocoWorldBase()
world.merge(arenas.empty_arena.EmptyArena())

robot = robots.Panda()
robot.add_gripper(grippers.gripper_factory("PandaGripper"))        # robosuite API :contentReference[oaicite:2]{index=2}
world.merge(robot)

ball = objects.BallObject("sphere", size=[0.04], rgba=[0, .5, .5, 1]).get_obj()
ball.set("pos", "0.5 0 0.04")
world.worldbody.append(ball)

model, data = world.get_model(mode="mujoco"), mujoco.MjData(world.get_model(mode="mujoco"))

viewer = launch_passive(model, data)                                # free camera GUI

# ---------------------------------------------------------------- helper
def save_viewer_frame(viewer, model, data, path):
    """
    Grab a PNG that matches the on-screen viewer image (camera, overlays, size).
    Grows the model’s off-screen framebuffer on-the-fly if the window is bigger.
    """
    with viewer.lock():                             # thread-safe
        w, h = viewer.viewport.width, viewer.viewport.height

        # --- ensure framebuffer ≥ window size (MuJoCo default is 640×480) ---
        if w > model.vis.global_.offwidth or h > model.vis.global_.offheight:
            model.vis.global_.offwidth  = max(w, model.vis.global_.offwidth)
            model.vis.global_.offheight = max(h, model.vis.global_.offheight)
        # --------------------------------------------------------------------
        renderer = mujoco.Renderer(model, width=w, height=h)
        renderer.update_scene(data,
                              camera=viewer.cam,
                              scene_option=viewer.opt)   # mirror GUI view
        Image.fromarray(renderer.render()).save(path)    # save PNG

# ---------------------------------------------------------------- run + dump
os.makedirs("viewer_frames", exist_ok=True)
step, every, act_dim = 0, 30, model.nu

while data.time < 10.0:                                             # 10-second demo
    data.ctrl[:] = np.random.uniform(-80, 80, act_dim)
    mujoco.mj_step(model, data)
    viewer.sync()

    if step % every == 0:
        fname = f"viewer_frames/frame_{step:06d}.png"
        save_viewer_frame(viewer, model, data, fname)
        print("saved", fname)
    step += 1

print(f"Finished – {step} sim steps; {(step//every)+1} PNGs in viewer_frames/")
