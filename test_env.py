import myenvs
import matplotlib.pyplot as plt
import gym
from collections import defaultdict
import numpy as np
import cv2
import os
import os.path as osp

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

_my_game_envs = defaultdict(set)
for env in myenvs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _my_game_envs[env_type].add(env.id)

def img2video(imgs, video_dir, fps):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
    img_size = imgs[0].shape[:2][::-1] # w and h
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for img in imgs:
        frame = img[:, :, ::-1] # RGB to BGR
        videoWriter.write(frame)

    videoWriter.release()
    print('Finish changing!')

env = myenvs.make("FetchThrowDice-v0")
rendered_imgs = []
for j in range(3):
    _ = env.reset()
    rendered_imgs.append(env.render("rgb_array"))
    cv2.imwrite("reset{:d}.png".format(j), rendered_imgs[-1])
    for i in range(50):
        action = (np.random.rand(8) - 0.5) * 2
        obs, reward, done, info = env.step(action)
        print(reward)
        print(obs["achieved_goal"])
        rendered_imgs.append(env.render("rgb_array"))


img2video(rendered_imgs, "./output.avi", 24)
