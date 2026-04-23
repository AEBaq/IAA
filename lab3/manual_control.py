#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.simulator import Simulator
from duckie_env import DuckiebotWrapper

parser = argparse.ArgumentParser()
args = parser.parse_args()

env = Simulator(
    map_name='iaa26_lab3',
    domain_rand=False,
    accept_start_angle_deg=4,
    full_transparency=True,
    distortion=False,
)

env = DuckiebotWrapper(env, 'map/iaa26_lab3_graph.yaml')
obs = env.reset()
env.render()



@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    # Action format: [velocity, steering]
    #   velocity ∈ [-1, 1]  (negative = reverse, for manual testing)
    #   steering ∈ [-1, 1]  (positive = turn left, negative = turn right)
    # The DuckiebotWrapper._vel_steer_to_wheels() converts to [left, right].

    velocity = 0.0
    steering = 0.0

    if key_handler[key.UP]:
        velocity += 0.44
    if key_handler[key.DOWN]:
        velocity -= 0.44
    if key_handler[key.LEFT]:
        steering += 1.0
    if key_handler[key.RIGHT]:
        steering -= 1.0
    if key_handler[key.SPACE]:
        velocity = 0.0
        steering = 0.0

    # Boost with shift (useful for testing)
    if key_handler[key.LSHIFT]:
        velocity *= 1.5

    # Clamp to valid ranges
    action = np.array([
        np.clip(velocity, -1.0, 1.0),
        np.clip(steering, -1.0, 1.0),
    ])

    obs, reward, done, info = env.step(action)

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs['image'])
        im.save("screen.png")

    if done:
        print("done!")
        obs = env.reset()
        env.render()
        return  # avoid double render

    env.render()  # on_draw fires here


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()
env.close()