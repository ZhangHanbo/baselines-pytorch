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
        while(True):
            if RENDER:
                env.render()
            action,distri = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            total_r += reward
            transition = torch.Tensor(np.hstack((observation, distri, action, reward, done, observation_)))
            agent.store_transition(transition)
            observation = observation_
            if done:
                break
            step += 1
        agent.learn()
        print('reward: ' + str(total_r) + ' episode: ' + str(episode))
        if total_r > 200:
            RENDER = True
    print('game over')
    env.close()

def run_ppo(env, agent, max_episode, step_episode):
    step = 0
    RENDER = False
    for episode in range(max_episode):
        memory_count = 0
        while(memory_count < agent.memory_size):
            total_r = 0
            observation = env.reset()
            while (True):
                if RENDER:
                    env.render()
                action, distri = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                total_r += reward
                transition = torch.Tensor(np.hstack((observation, distri, action, reward, done, observation_)))
                agent.store_transition(transition)
                memory_count += 1
                observation = observation_
                if done:
                    break
                step += 1
            print('reward: ' + str(total_r) + ' episode: ' + str(episode))
        agent.learn()
        if total_r > 2000:
            RENDER = True
    print('game over')
    env.close()


'''
if done and t < 199:
    reward = 200 - t
else:
    reward = 10 * np.abs(observation_[1])
'''

if __name__ == "__main__":
    # maze game
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    n_features = env.observation_space.shape[0]
    if env.action_space.shape == ():
        n_actions = env.action_space.n # decrete action space, value based rl brain
    else:
        n_actions = env.action_space.shape[0]


    DQNconfig = {
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

    PGconfig = {
        'n_features': n_features,
        'n_actions': n_actions,
        'lr': 0.01,
        'memory_size': 3000,
        'reward_decay': 0.995,
        'batch_size': 1024,
        'GAE_lambda': 0.97,
        'value_type' : 'FC',
        'optimizer': optim.Adam
    }

    RL_brain = DRL_Agent.AdaptiveKLPPO_Softmax(PGconfig)
    run_ppo(env, RL_brain, 3000, 200)
