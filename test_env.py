import numpy as np
import cv2
import os
import os.path as osp
import argparse
from collections import defaultdict

import gym
import myenvs

import sys
sys.path.append("./softgymenvs/")
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

sys.path.append("./ravensenvs/")


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

def main_my_own():
    env = myenvs.make("FetchThrow-v0")
    rendered_imgs = []
    for j in range(3):
        _ = env.reset()
        rendered_imgs.append(env.render("rgb_array"))
        cv2.imwrite("reset{:d}.png".format(j), rendered_imgs[-1])
        for i in range(50):
            action = (np.random.rand(4) - 0.5) * 2
            obs, reward, done, info = env.step(action)
            print(reward)
            print(obs["achieved_goal"])
            rendered_imgs.append(env.render("rgb_array"))

    img2video(rendered_imgs, "./FetchThrow.avi", 24)

def main_softgym():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    env_name = "RopeFlatten"
    save_video_dir = './data/'
    env_kwargs = env_arg_dict[env_name]
    img_size = 256
    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = 1
    env_kwargs['render'] = True
    env_kwargs['headless'] = True
    env_kwargs['observation_mode'] = "key_point"

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))

    frames = []
    for j in range(5):
        env.reset()
        frames.append(env.get_image(img_size, img_size))
        for i in range(env.horizon):
            action = env.action_space.sample()
            # action = np.zeros(env.action_space.shape)
            # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
            # intermediate frames. Only use this option for visualization as it increases computation.
            obs, rew, _, info = env.step(action, record_continuous_video=True, img_size=img_size)
            # frames.extend(info['flex_env_recorded_frames'])
            frames.append(env.get_image(img_size, img_size))
    if save_video_dir is not None:
        if not osp.exists(save_video_dir):
            os.makedirs(save_video_dir)
        save_name = osp.join(".", env_name + '.avi')
        img2video(frames, save_name, 24)
        print('Video generated and save to {}'.format(save_name))

def main_my_softgym():
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    env = myenvs.make("RopeConfiguration-v0")
    img_size = 256

    frames = []
    for j in range(20):
        env.reset()
        frames.append(env.get_image(img_size, img_size))
        achieved_goals = []
        for i in range(50):
            # action = np.zeros(env.action_space.shape)
            action = env.action_space.sample()
            # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
            # intermediate frames. Only use this option for visualization as it increases computation.
            obs, rew, _, info = env.step(action)
            # frames.extend(info['flex_env_recorded_frames'])
            frames.append(env.get_image(img_size, img_size))
            # print(obs["observation"].reshape(-1 ,3))
            achieved_goals.append(np.expand_dims(obs["achieved_goal"], axis=0))
        achieved_goals = np.concatenate(achieved_goals, axis = 0)
        print(env.compute_reward(
            achieved_goal=achieved_goals,
            desired_goal=np.tile(np.expand_dims(achieved_goals[0], axis=0), (achieved_goals.shape[0], 1)),
            info=None
        ))

    save_name = osp.join(".", 'RopeConfiguration.avi')
    img2video(frames, save_name, 24)
    print('Video generated and save to {}'.format(save_name))

main_my_softgym()
