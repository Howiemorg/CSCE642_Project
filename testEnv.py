"""Spawn a single drone, then command it to go to two setpoints consecutively, and plots the xyz output."""

import matplotlib.pyplot as plt
import numpy as np

from PyFlyt.core import Aviary, loadOBJ, obj_collision, obj_visual

# initialize the log
log = np.zeros((1000, 3), dtype=np.float32)

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])
options=dict(use_camera=True)

# environment setup
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx", drone_options=options)


# set to position control
env.set_mode(7)

# load the visual and collision entities and load the duck
visualId = obj_visual(env, "duck.obj")
collisionId = obj_collision(env, "duck.obj")
loadOBJ(
    env,
    visualId=visualId,
    collisionId=collisionId,
    baseMass=1.0,
    basePosition=[0.0, 0.0, 10.0],
)

# call this to register all new bodies for collision
env.register_all_new_bodies()

# for the first 500 steps, go to x=1, y=0, z=1
setpoint = np.array([1.0, 0.0, 0.0, 1.0])
env.set_setpoint(0, setpoint)

for i in range(500):
    env.step()

    # record the linear position state
    log[i] = env.state(0)[-1]

# for the next 500 steps, go to x=0, y=0, z=2, rotate 45 degrees
setpoint = np.array([0.0, 0.0, np.pi / 4, 2.0])
env.set_setpoint(0, setpoint)

for i in range(500, 1000):
    env.step()

    # record the linear position state
    log[i] = env.state(0)[-1]

# plot stuff out
plt.plot(np.arange(1000), log)
plt.show()

# get the camera image and show it
RGBA_img = env.drones[0].rgbaImg
DEPTH_img = env.drones[0].depthImg
SEG_img = env.drones[0].segImg

plt.imshow(RGBA_img)
# plt.show()
plt.imshow(DEPTH_img)
# plt.show()
plt.imshow(SEG_img)
plt.show()