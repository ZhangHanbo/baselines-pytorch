import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import Feature_Extractor
import DRL_Agent
import gym
from Evironment.maze_env import Maze

def run(env, agent, max_episode, step_episode):
    step = 0
    RENDER = False
    for episode in range(max_episode):
        total_r = 0
        observation = env.reset()
        for t in range(step_episode):
            if RENDER:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if done and t < 199:
                reward = 200 - t
            else:
                reward = 10 * np.abs(observation_[1])

            total_r += reward
            agent.store_transition(observation, action, reward, observation_, done)
            if step > 200:
                agent.learn()
            observation = observation_
            if done:
                break
            step += 1
        print('reward: ' + str(total_r) + ' episode: ' + str(episode))
        if total_r > 100:
            RENDER = True
    print('game over')
    env.close()

if __name__ == "__main__":
    # maze game
    env = gym.make('MountainCar-v0')
    n_features = env.observation_space.shape[0]
    if env.action_space.shape == ():
        n_actions = env.action_space.n # decrete action space, value based rl brain
    else:
        n_actions = env.action_space.shape[0]

    config = {
        'n_features':n_features,
        'n_actions':n_actions,
        'lr':0.001,
        'mom':0,
        'reward_decay':0.9,
        'e_greedy':1,
        'replace_target_iter':600,
        'memory_size':10000,
        'batch_size':64,
        'e_greedy_increment':0.001,
        'optimizer': optim.RMSprop

    }

    RL_brain = DRL_Agent.DQN(config)
    run(env, RL_brain, 200, 600)
