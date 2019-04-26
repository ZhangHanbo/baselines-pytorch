import argparse
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np

import gym
from gym import wrappers
from envs.maze_env import Maze

import basenets
import agents
import configs
from utils import hindsight

from tensorboardX import SummaryWriter
from matplotlib import pyplot

torch.set_default_tensor_type(torch.FloatTensor)


DICRETE_ACTION_SPACE_LIST = ("CartPole-v1",)
CONTINUOUS_ACTION_SPACE_LIST = ("Pendulum-v0", "Reacher-v1", "BaxterReacher-v0")

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

def run_DQN_hindsight(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    if env['name'] not in DICRETE_ACTION_SPACE_LIST:
        raise RuntimeError("DQN is used for games with dicrete action space.")
    step = 0
    total_r = 0
    for episode in range(max_episode):
        observation = env['env'].reset()
        t = 0
        transition = {
            'state': [],
            'action': [],
            'distr': [],
            'reward': [],
            'next_state': [],
            'done': [],
        }
        while(t < step_episode):
            # visualization
            if isrender:
                env['env'].render()
            # take one step
            action, distri = agent.choose_action(observation)
            observation_, reward, done, info = env['env'].step(action)
            reward = modify_reward(observation, action, reward, observation_, env)
            # store transition
            transition['state'].append([observation] if len(np.shape(observation)) == 0 else observation)
            transition['action'].append([action] if len(np.shape(action)) == 0 else action)
            transition['distr'].append([distri] if len(np.shape(distri)) == 0 else distri)
            transition['reward'].append([reward])
            transition['next_state'].append([observation_] if len(np.shape(observation_)) == 0 else observation_)
            transition['done'].append([done])
            t += 1
            if done:
                break
            observation = observation_

        if agent.hindsight_replay:
            # TODO: implement hindsight replay buffer
            h_transition = hindsight.hindsight_transition(transition, info=None)
            for key in transition.keys():
                transition[key] += h_transition[key]
        # make transition into numpy array
        for key in transition.keys():
            transition[key] = np.array(transition[key])
        agent.store_transition(transition)
        # training
        episode_lenth = transition['state'].shape[0]
        step += episode_lenth
        total_r += transition['reward'].sum()

        if not args.test:
            for _ in range(episode_lenth):
                # training
                if step > agent.memory.batch_size:
                    for i in range(agent.update_num_per_step):
                        agent.learn()
            agent.episode_counter += 1
            if (episode + 1) % agent.snapshot_episode == 0:
                agent.save_model(output_dir)

        if (episode + 1) % args.display == 0:
            print('reward: ' + str(total_r / args.display) + ' episode: ' + str(episode + 1))
            logger.add_scalar("reward/train", total_r / args.display, episode)
            if renderth is not None and total_r / args.display > renderth:
                isrender = True
            total_r = 0

    env['env'].close()

def run_DQN(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    if env['name'] not in DICRETE_ACTION_SPACE_LIST:
        raise RuntimeError("DQN is used for games with dicrete action space.")
    step = 0
    total_r = 0
    for episode in range(max_episode):
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
            if step > agent.batch_size:
                for i in range(agent.update_num_per_step):
                    agent.learn()
            step += 1
            t += 1
            # if the episode is done
            if done:
                break
            # prepare for next step
            observation = observation_

        if (episode + 1) % args.display == 0:
            print('reward: ' + str(total_r / args.display) + ' episode: ' + str(episode + 1))
            logger.add_scalar("reward/train", total_r / args.display, episode)
            total_r = 0

        agent.episode_counter += 1
        if (episode + 1) % agent.snapshot_episode == 0:
            agent.save_model(output_dir)

        # if reward of the episode is higher than the threshold, visualize the training process
        if renderth is not None and total_r > renderth:
            isrender = True
    env['env'].close()

def run_PG(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    step = 0
    total_r = 0
    for i_episode in range(max_episode):
        observation = env['env'].reset()
        t = 0
        transition = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'done': [],
        }
        if env['name'] in CONTINUOUS_ACTION_SPACE_LIST:
            transition['mu'] = []
            transition['sigma'] = []
        elif env['name'] in DICRETE_ACTION_SPACE_LIST:
            transition['distr'] = []

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
            # store transition
            transition['state'].append([observation] if len(np.shape(observation)) == 0 else observation)
            transition['action'].append([action] if len(np.shape(action)) == 0 else action)
            transition['reward'].append([reward])
            transition['next_state'].append([observation_] if len(np.shape(observation_)) == 0 else observation_)
            transition['done'].append([done])
            if env['name'] in CONTINUOUS_ACTION_SPACE_LIST:
                transition['mu'].append([mu] if len(np.shape(mu)) == 0 else mu)
                transition['sigma'].append([sigma] if len(np.shape(sigma)) == 0 else sigma)
            elif env['name'] in DICRETE_ACTION_SPACE_LIST:
                transition['distr'].append([distri] if len(np.shape(distri)) == 0 else distri)
            t += 1
            if done:
                break
            observation = observation_

        if agent.hindsight_replay:
            # TODO: implement hindsight replay buffer
            h_transition = hindsight.hindsight_transition(transition, info=None)
            for key in transition.keys():
                transition[key] += h_transition[key]
        # make transition into numpy array
        for key in transition.keys():
            transition[key] = np.array(transition[key])
        agent.store_transition(transition)
        # training
        episode_lenth = transition['state'].shape[0]
        step += episode_lenth
        total_r += transition['reward'].sum()

        if (i_episode + 1) % args.display == 0:
            print('episode: ' + str(i_episode + 1) + '   reward: ' + str(total_r / args.display))
            logger.add_scalar("reward/train", total_r / args.display, i_episode)
            if renderth is not None and total_r / args.display > renderth:
                isrender = True
            total_r = 0

        if not args.test:
            if step >= agent.memory.max_size:
                for i in range(agent.update_num_per_step):
                    agent.learn()
                step = 0
            agent.episode_counter += 1
            if (i_episode + 1) % agent.snapshot_episode == 0:
                agent.save_model(output_dir)
    env['env'].close()

def run_DPG_hindsight(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    if env['name'] not in CONTINUOUS_ACTION_SPACE_LIST:
        raise RuntimeError("DPG is used for games with continuous action space.")
    step = 0
    total_r = 0
    for i_episode in range(max_episode):
        # collect transitions
        observation = env['env'].reset()
        t = 0
        transition = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'done': [],
        }
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
            # store transition
            transition['state'].append([observation] if len(np.shape(observation)) == 0 else observation)
            transition['action'].append([action] if len(np.shape(action)) == 0 else action)
            transition['reward'].append([reward])
            transition['next_state'].append([observation_] if len(np.shape(observation_)) == 0 else observation_)
            transition['done'].append([done])
            t += 1
            if done:
                break
            observation = observation_

        if agent.hindsight_replay:
            # TODO: implement hindsight replay buffer
            h_transition = hindsight.hindsight_transition(transition, info = None)
            for key in transition.keys():
                transition[key] += h_transition[key]
        # make transition into numpy array
        for key in transition.keys():
            transition[key] = np.array(transition[key])
        agent.store_transition(transition)
        # training
        episode_lenth = transition['state'].shape[0]
        step += episode_lenth
        total_r += transition['reward'].sum()

        if not args.test:
            for _ in range(episode_lenth):
                if step >= agent.batch_size:
                    for i in range(agent.update_num_per_step):
                        agent.learn()
            agent.episode_counter += 1
            if (i_episode + 1) % agent.snapshot_episode == 0:
                agent.save_model(output_dir)

        if (i_episode + 1) % args.display == 0:
            print('episode: ' + str(i_episode + 1) + '   reward: ' + str(total_r / args.display))
            logger.add_scalar("reward/train", total_r / args.display, i_episode)
            if renderth is not None and total_r / args.display > renderth:
                isrender = True
            total_r = 0

    env['env'].close()

def run_DPG(env, agent, max_episode, step_episode = np.inf, isrender = False, renderth = None):
    if env['name'] not in CONTINUOUS_ACTION_SPACE_LIST:
        raise RuntimeError("DPG is used for games with continuous action space.")
    step = 0
    total_r = 0
    for i_episode in range(max_episode):
        observation = env['env'].reset()
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
            if not args.test:
                if step >= agent.batch_size:
                    for i in range(agent.update_num_per_step):
                        agent.learn()
            step += 1
            t += 1
            if done:
                break
            # swap observation
            observation = observation_

        if not args.test:
            agent.episode_counter += 1
            if (i_episode + 1) % agent.snapshot_episode == 0:
                agent.save_model(output_dir)

        if (i_episode + 1) % args.display == 0:
            print('episode: ' + str(i_episode + 1) + '   reward: ' + str(total_r / args.display))
            logger.add_scalar("reward/train", total_r / args.display, i_episode)
            if renderth is not None and total_r / args.display > renderth:
                isrender = True
            total_r = 0
    env['env'].close()

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--algo', default='NAF',
                        help='algorithm to use: DQN | DDQN | DuelingDQN | DDPG | NAF | PG | NPG | TRPO | PPO')
    parser.add_argument('--env', default="Pendulum-v0",
                        help='name of the environment to run')
    parser.add_argument('--ou_noise', type=bool, default=True)
    # TODO: SUPPORT PARAM NOISE
    parser.add_argument('--param_noise', type=bool, default=False)
    # TODO: SUPPORT NOISE END
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--display', type=int, default=5, metavar='N',
                        help='episode interval for display (default: 5)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='whether to resume training from a specific checkpoint')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='resume from this checkpoint')
    parser.add_argument('--render', type=float, default=np.inf,
                        help='when to render GUI (default: 0). WARNING: this is the episode return threshold which '
                             'controls when to render a GUI window, therefore, it should be set carefully with dif-'
                             'ferent environments and it will slow down the training process.')
    parser.add_argument('--test', help='test the specific policy.', action='store_true', default = False)
    parser.add_argument('--cpu', help='whether use cpu to train', action='store_true', default = False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()

    if args.resume or args.test and args.checkpoint == 0:
        raise RuntimeError("Checkpoint need to be specified.")

    RENDER = False
    if args.test:
        args.resume = True
        RENDER = True
        args.display = 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # maze game
    env = {}
    env['name'] = args.env
    if args.env == 'BaxterReacher-v0':
        from envs.BaxterReacher_v0 import Baxter as BaxterReacher_v0
        env['env'] = BaxterReacher_v0()
        env['env'].unwrapped()
    else:
        env['env'] = gym.make(env['name'])
        env['env'].seed(args.seed)
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

    logger = SummaryWriter(comment = args.algo + "-" + args.env)
    output_dir = os.path.join("output", "models", args.algo, args.env)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.algo in ("DQN", "DDQN", "DuelingDQN"):
        RL_brain = eval("agents." + args.algo + "(configs." + args.algo + "_" +
                        "".join(args.env.split("-")) + "." + args.algo + "config)")
        if args.resume:
            RL_brain.load_model(load_path=output_dir, load_point=args.checkpoint)
        if not args.cpu:
            RL_brain.cuda()
        run_DQN(env, RL_brain, max_episode=RL_brain.max_num_episode, step_episode=RL_brain.max_num_step,
                isrender=RENDER, renderth=args.render)
    elif args.algo in ("PG", "NPG", "TRPO", "PPO"):
        if DICRETE_ACTION_SPACE:
            RL_brain = eval("agents." + args.algo + "_Softmax(configs." + args.algo + "_" +
                            "".join(args.env.split("-")) + "." + args.algo + "config)")
        else:
            RL_brain = eval("agents." + args.algo + "_Gaussian(configs." + args.algo + "_" +
                            "".join(args.env.split("-")) + "." + args.algo + "config)")
        if args.resume:
            RL_brain.load_model(load_path=output_dir, load_point=args.checkpoint)
        if not args.cpu:
            RL_brain.cuda()
        run_PG(env, RL_brain, max_episode=RL_brain.max_num_episode, step_episode=RL_brain.max_num_step,
               isrender=RENDER, renderth=args.render)
    elif args.algo in ("DDPG", "NAF"):
        RL_brain = eval("agents." + args.algo + "(configs." + args.algo + "_" +
                        "".join(args.env.split("-")) + "." + args.algo + "config)")
        if args.resume:
            RL_brain.load_model(load_path=output_dir, load_point=args.checkpoint)
        if not args.cpu:
            RL_brain.cuda()
        run_DPG(env, RL_brain, max_episode=RL_brain.max_num_episode, step_episode=RL_brain.max_num_step,
                isrender=RENDER, renderth=args.render)
    logger.close()