import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import basenets
import agents
import gym
from envs.maze_env import Maze

def run_DQN(env, agent, max_episode, step_episode):
    step = 0
    RENDER = False
    for episode in range(max_episode):
        total_r = 0
        observation = env.reset()
        for i in range(step_episode):
            if RENDER:
                env.render()
            action, distri = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            total_r += reward
            transition = {
                'state': np.expand_dims(observation, 0),
                'action': np.expand_dims(np.array([action]), 0),
                'distr': np.expand_dims(distri, 0) if len(distri.shape) == 1 else distri,
                'reward': np.expand_dims(np.array([reward]), 0),
                'next_state': np.expand_dims(observation_, 0),
                'done': np.expand_dims(np.array([done]), 0),
            }
            agent.store_transition(transition)
            observation = observation_
            if done:
                break
            step += 1
            if step > 1000:
                agent.learn()
        print('reward: ' + str(total_r) + ' episode: ' + str(episode))

        # visualize training
        # if total_r > 200:
        #     RENDER = True

    print('game over')
    env.close()

def run_PG(env, agent, max_episode, step_episode):
    step = 0
    RENDER = False
    for episode in range(max_episode):
        total_r = 0
        observation = env.reset()
        for i in range(5000):
            if RENDER:
                env.render()
            action, distri = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            total_r += reward

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            transition = {
                'state': np.expand_dims(observation, 0),
                'action': np.expand_dims(np.array([action]), 0),
                'distr': np.expand_dims(distri, 0),
                'reward': np.expand_dims(np.array([reward]), 0),
                'next_state': np.expand_dims(observation_, 0),
                'done': np.expand_dims(np.array([done]), 0),
            }

            agent.store_transition(transition)
            observation = observation_
            if done:
                break
            step += 1

        print('reward: ' + str(total_r) + ' episode: ' + str(episode))
        if step>2000:
            agent.learn()

        # visualize training
        # if total_r > 200:
        #     RENDER = True

    print('game over')
    env.close()

def run_ppo(env, agent, max_episode, step_episode):
    step = 0
    RENDER = False
    for episode in range(max_episode):
        memory_count = 0
        episode_lenth = 300
        while(memory_count < agent.memory_size):
            total_r = 0
            observation = env.reset()
            while(True):
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
        # if total_r > -100:
        #    RENDER = True
    print('game over')
    env.close()

if __name__ == "__main__":
    # maze game
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    n_features = env.observation_space.shape[0]
    if env.action_space.shape == ():
        n_actions = env.action_space.n # decrete action space, value based rl brain
        DICRETE_ACTION_SPACE = True
    else:
        n_actions = env.action_space.shape[0]
        DICRETE_ACTION_SPACE = False

    DQNconfig = {
        'n_states':n_features,
        'dicrete_action': DICRETE_ACTION_SPACE,
        'n_actions':n_actions,
        'n_action_dims': 1,
        'lr':0.001,
        'mom':0.9,
        'reward_decay':0.9,
        'e_greedy':0.9,
        'replace_target_iter':100,
        'memory_size': 2000,
        'batch_size': 32,
        'e_greedy_increment':None,
        'optimizer': optim.RMSprop

    }

    PGconfig = {
        'n_states': n_features,
        'n_actions': n_actions,
        'n_action_dims': 1,
        'lr': 3e-4,
        'memory_size': 500,
        'reward_decay': 0.995,
        'batch_size': 500,
        'GAE_lambda': 0.97,
        'value_type' : 'FC',
        'optimizer': optim.Adam
    }

    RL_brain = agents.DQN(DQNconfig)
    run_DQN(env, RL_brain, 10000, 500)
