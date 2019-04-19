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
import configs.DDPG_Pendulumv0


DICRETE_ACTION_SPACE_LIST = ("CartPole-v1",)
CONTINUOUS_ACTION_SPACE_LIST = ("Pendulum-v0",)

def modify_reward(s, a, r, s_, env):
    # modify the reward for training
    if env['name'] == 'CartPole-v1':
        x, x_dot, theta, theta_dot = s_
        r1 = (env['env'].x_threshold - abs(x)) / env['env'].x_threshold - 0.8
        r2 = (env['env'].theta_threshold_radians - abs(theta)) \
             / env['env'].theta_threshold_radians - 0.5
        return r1 + r2
    else:
        return r

def run_DQN(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    if env['name'] not in DICRETE_ACTION_SPACE_LIST:
        raise RuntimeError("DQN is used for games with dicrete action space.")
    step = 0
    for episode in range(max_episode):
        total_r = 0
        observation = env['env'].reset()
        t = 0
        while(t < step_episode):
            # visualization
            if isrender:
                env['env'].render()
            # take one step
            action, distri = agent.choose_action(observation)
            observation_, reward, done, info = env['env'].step(action)
            total_r += reward
            reward = modify_reward(observation, action, reward, observation_, env)
            # store transition
            transition = {
                'state': np.expand_dims(observation, 0),
                'action': np.expand_dims([action] if len(np.shape(action)) == 0 else action, 0),
                'distr': np.expand_dims(distri, 0) if len(distri.shape) == 1 else distri,
                'reward': np.expand_dims([reward], 0),
                'next_state': np.expand_dims(observation_, 0),
                'done': np.expand_dims([done], 0),
            }
            agent.store_transition(transition)
            # training
            if step > agent.memory.max_size:
                agent.learn()
            step += 1
            t += 1
            # if the episode is done
            if done:
                break
            # prepare for next step
            observation = observation_
        print('reward: ' + str(total_r) + ' episode: ' + str(episode))
        # if reward of the episode is higher than the threshold, visualize the training process
        if renderth is not None and total_r > renderth:
            isrender = True
    print('game over')
    env['env'].close()

def run_PG(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    step = 0
    for i_episode in range(max_episode):
        observation = env['env'].reset()
        total_r = 0
        t = 0
        while (t < step_episode):
            if isrender:
                env['env'].render()
            # choose action
            if env['name'] in CONTINUOUS_ACTION_SPACE_LIST:
                action, mu ,sigma = agent.choose_action(observation)
                action = np.clip(action, env['env'].action_space.low, env['env'].action_space.high)
            elif env['name'] in DICRETE_ACTION_SPACE_LIST:
                action, distri = agent.choose_action(observation)
            # transition
            observation_, reward, done, info = env['env'].step(action)
            reward = modify_reward(observation, action, reward, observation_, env)
            # for policy gradient methods, "done" flag for every episode is needed for training.
            if t == step_episode - 1:
                done = True
            total_r = total_r + reward
            # store transition
            transition = {
                'state': np.expand_dims(observation, 0),
                'action': np.expand_dims([action] if len(np.shape(action)) == 0 else action, 0),
                'reward': np.expand_dims([reward / 10], 0),
                'next_state': np.expand_dims(observation_, 0),
                'done': np.expand_dims([done], 0),
            }
            if env['name'] in CONTINUOUS_ACTION_SPACE_LIST:
                transition['mu'] = np.expand_dims(mu, 0)
                transition['sigma'] = np.expand_dims(sigma, 0)
            elif env['name'] in DICRETE_ACTION_SPACE_LIST:
                transition['distr'] = np.expand_dims(distri, 0) if len(distri.shape) == 1 else distri
            agent.store_transition(transition)
            step += 1
            t += 1
            if done:
                break
            # swap observation
            observation = observation_
        print('episode: ' + str(i_episode) + '   reward: ' + str(total_r))
        print("Episode finished after {} timesteps".format(t))
        if step >= agent.memory.max_size:
            agent.learn()
            step = 0
        if renderth is not None and total_r > renderth:
            isrender = True

def run_DPG(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    if env['name'] not in CONTINUOUS_ACTION_SPACE_LIST:
        raise RuntimeError("DPG is used for games with continuous action space.")
    step = 0
    for i_episode in range(max_episode):
        observation = env['env'].reset()
        total_r = 0
        t = 0
        while (t < step_episode):
            if isrender:
                env['env'].render()
            # choose action
            action = agent.choose_action(observation)
            action = np.clip(action, env['env'].action_space.low, env['env'].action_space.high)
            # transition
            observation_, reward, done, info = env['env'].step(action)
            reward = modify_reward(observation, action, reward, observation_, env)
            # for policy gradient methods, "done" flag for every episode is needed for training.
            if t == step_episode - 1:
                done = True
            total_r = total_r + reward
            # store transition
            transition = {
                'state': np.expand_dims(observation, 0),
                'action': np.expand_dims([action] if len(np.shape(action)) == 0 else action, 0),
                'reward': np.expand_dims([reward / 10], 0),
                'next_state': np.expand_dims(observation_, 0),
                'done': np.expand_dims([done], 0),
            }
            agent.store_transition(transition)
            step += 1
            t += 1
            if done:
                break
            # swap observation
            observation = observation_
        print('episode: ' + str(i_episode) + '   reward: ' + str(total_r))
        print("Episode finished after {} timesteps".format(t))
        if step >= agent.memory.max_size:
            agent.learn()
        if renderth is not None and total_r > renderth:
            isrender = True

if __name__ == "__main__":
    # maze game
    env = {}
    env['name'] = "Pendulum-v0"
    # env['name'] = "CartPole-v1"
    env['env'] = gym.make(env['name'])
    env['env'] = env['env'].unwrapped
    n_features = env['env'].observation_space.shape[0]
    if env['name'] in DICRETE_ACTION_SPACE_LIST:
        n_actions = env['env'].action_space.n # decrete action space, value based rl brain
        n_action_dims = 1
        DICRETE_ACTION_SPACE = True
    elif env['name'] in CONTINUOUS_ACTION_SPACE_LIST:
        n_actions = None
        n_action_dims = env['env'].action_space.shape[0]
        DICRETE_ACTION_SPACE = False
    else:
        raise RuntimeError("Game not defined as dicrete or continuous.")

    DQNconfig = {
        'n_states':n_features,
        'dicrete_action': DICRETE_ACTION_SPACE,
        'n_actions':n_actions,
        'n_action_dims': 1,
        'lr':0.01,
        'mom':0,
        'reward_decay':0.9,
        'e_greedy':0.9,
        'replace_target_iter':100,
        'memory_size': 2000,
        'batch_size': 32,
        'e_greedy_increment':None,
        'optimizer': optim.Adam
    }

    PGconfig = {
        'n_states': n_features,
        'n_actions': n_actions,
        'n_action_dims': 1,
        'lr': 3e-3,
        'memory_size': 500,
        'reward_decay': 0.995,
        'batch_size': 500,
        'GAE_lambda': 0.97,
        'value_type' : 'FC',
        'optimizer': optim.Adam
    }

    PPOconfig_dicrete = {
        'n_states': env['env'].observation_space.shape[0],
        'n_action_dims': 1,
        # 'action_bounds': env['env'].action_space.high,
        'memory_size':600,
        'reward_decay':0.95,
        'steps_per_update':15,
        'batch_size':3000,
        'max_grad_norm': 2,
        'GAE_lambda':0.95,
        'clip_epsilon': 0.2,
        'lr' : 3e-2,
        'lr_v':3e-2,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC'
    }

    PPOconfig_continuous = {
        'n_states': env['env'].observation_space.shape[0],
        'n_action_dims': 1,
        # 'action_bounds': env['env'].action_space.high,
        'memory_size': 600,
        'reward_decay': 0.95,
        'steps_per_update': 15,
        'batch_size': 3000,
        'max_grad_norm': 2,
        'GAE_lambda': 0.95,
        'clip_epsilon': 0.2,
        'lr': 3e-2,
        'lr_v': 3e-2,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC'
    }

    # RL_brain = agents.TRPO_Gaussian(TRPOconfig)
    RL_brain = agents.DDPG(configs.DDPG_Pendulumv0.DDPGconfig)
    # run_PG(env, RL_brain, max_episode = 1000, step_episode= 1000)
    run_DPG(env, RL_brain, max_episode=1000, step_episode=1000)
    # test git