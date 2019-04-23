import argparse
import os

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

from tensorboardX import SummaryWriter
from matplotlib import pyplot


DICRETE_ACTION_SPACE_LIST = ("CartPole-v1",)
CONTINUOUS_ACTION_SPACE_LIST = ("Pendulum-v0", "Reacher-v1")

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
            if step > agent.memory.max_size:
                for i in range(args.updates_per_step):
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
        if (episode + 1) % args.snapshot_episode == 0:
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

        if (i_episode + 1) % args.display == 0:
            print('episode: ' + str(i_episode + 1) + '   reward: ' + str(total_r / args.display))
            logger.add_scalar("reward/train", total_r / args.display, i_episode)
            total_r = 0

        agent.episode_counter += 1
        if (i_episode + 1) % args.snapshot_episode == 0:
            agent.save_model(output_dir)

        if step >= agent.memory.max_size:
            for i in range(args.updates_per_step):
                agent.learn()
            step = 0

        if renderth is not None and total_r > renderth:
            isrender = True

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
            if step >= agent.memory.max_size:
                for i in range(args.updates_per_step):
                    agent.learn()
            step += 1
            t += 1
            if done:
                break
            # swap observation
            observation = observation_

        if (i_episode + 1) % args.display == 0:
            print('episode: ' + str(i_episode + 1) + '   reward: ' + str(total_r / args.display))
            logger.add_scalar("reward/train", total_r / args.display, i_episode)
            if renderth is not None and total_r / args.display > renderth:
                isrender = True
            total_r = 0

        agent.episode_counter += 1
        if (i_episode + 1) % args.snapshot_episode == 0:
            agent.save_model(output_dir)

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
    parser.add_argument('--num_steps', type=int, default=200, metavar='N',
                        help='max episode length (default: 200)')
    parser.add_argument('--num_episodes', type=int, default=500, metavar='N',
                        help='number of episodes (default: 500)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--display', type=int, default=5, metavar='N',
                        help='episode interval for display (default: 5)')
    parser.add_argument('--snapshot_episode', type=int, default=100, metavar='N',
                        help='snapshot interval (default: 100)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='whether to resume training from a specific checkpoint')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='resume from this checkpoint')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # maze game
    env = {}
    env['name'] = args.env
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
    output_dir = os.path.join("output", "models", args.algo)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.algo in ("DQN", "DDQN", "DuelingDQN"):
        RL_brain = eval("agents." + args.algo + "(configs." + args.algo + "_" +
                        "".join(args.env.split("-")) + "." + args.algo + "config)")
        if args.resume:
            RL_brain.load_model(load_path=output_dir, load_point=args.checkpoint)
        run_DQN(env, RL_brain, max_episode=args.num_episodes, step_episode=args.num_steps)
    elif args.algo in ("PG", "NPG", "TRPO", "PPO"):
        if DICRETE_ACTION_SPACE:
            RL_brain = eval("agents." + args.algo + "_Softmax(configs." + args.algo + "_" +
                            "".join(args.env.split("-")) + "." + args.algo + "config)")
        else:
            RL_brain = eval("agents." + args.algo + "_Gaussian(configs." + args.algo + "_" +
                            "".join(args.env.split("-")) + "." + args.algo + "config)")
        if args.resume:
            RL_brain.load_model(load_path=output_dir, load_point=args.checkpoint)
        run_PG(env, RL_brain, max_episode=args.num_episodes, step_episode=args.num_steps)
    elif args.algo in ("DDPG", "NAF"):
        RL_brain = eval("agents." + args.algo + "(configs." + args.algo + "_" +
                        "".join(args.env.split("-")) + "." + args.algo + "config)")
        if args.resume:
            RL_brain.load_model(load_path=output_dir, load_point=args.checkpoint)
        run_DPG(env, RL_brain, max_episode=args.num_episodes, step_episode=args.num_steps)
    logger.close()