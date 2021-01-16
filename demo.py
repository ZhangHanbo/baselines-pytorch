import argparse
import os
import torch
import numpy as np
from gym import spaces
from tensorboardX import SummaryWriter
from collections import deque

import sys
sys.path.append("./softgymenvs/")

from utils.envbuilder import build_env, set_global_seeds
from utils.vec_envs import space_dim
from agents import *
from agents.config import *
from configs import CONFIGS

torch.set_default_tensor_type(torch.FloatTensor)

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--alg', default='NAF',
                        help='algorithm to use: DQN | DDQN | DuelingDQN | DDPG | NAF | PG | NPG | TRPO | PPO')
    parser.add_argument('--env', default="Reacher-v1",
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--num_envs', type=int, default=1, metavar='N',
                        help='env numbers (default: 1)')
    parser.add_argument('--num_evals', type=int, default=10, metavar='N',
                        help='evaluation episode number each time (default: 10)')
    parser.add_argument('--unnormobs', action='store_true', default=False,
                        help='whether to normalize inputs')
    parser.add_argument('--unnormret', action='store_true', default=False,
                        help='whether to normalize outputs')
    parser.add_argument('--unnormact', action='store_true', default=False,
                        help='whether to normalize outputs')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='resume from this checkpoint')
    parser.add_argument('--render', action='store_true', default=False,
                        help='whether to render GUI (default: False) during evaluation.')
    parser.add_argument('--greedy', action='store_true', default=False,
                        help='whether to render GUI (default: False) during evaluation.')
    parser.add_argument('--cpu', help='whether use cpu to train', action='store_true', default = False)
    parser.add_argument('--usedemo', action='store_true', default=False,
                        help='whether to use imitation learning to improve performance')
    parser.add_argument('--demopath', default='demos',
                        help='path where you stores the demonstration file demo.hdf5. Now only HTRPO supported.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    args.reward = "sparse"

    configs = {
        "norm_ob": not args.unnormobs,
        "norm_rw": not args.unnormret,
    }
    # TODO: REMOVE THE DISABLING OF NORMALIZATION FOR TD3, NAF and DDPG.
    #  IMPROVEMENTS SHOULD BE MADE FOR THESE ALGORITHMS.

    # build game environment
    env, env_type, env_id = build_env(args)
    env.env_id = env_id
    env.env_type = env_type
    env.alg = args.alg
    env_obs_space = env.observation_space
    env_act_space = env.action_space
    n_states = space_dim(env_obs_space)

    if isinstance(env_act_space, spaces.Discrete):
        n_actions = env_act_space.n  # decrete action space, value based rl brain
        n_action_dims = 1
        DICRETE_ACTION_SPACE = True
    elif isinstance(env_act_space, spaces.Box):
        n_actions = None
        n_action_dims = env_act_space.shape[0]
        DICRETE_ACTION_SPACE = False
    elif isinstance(env_act_space, np.ndarray):
        n_actions = len(env_act_space)
        n_action_dims = 1
        DICRETE_ACTION_SPACE = True
    else:
        assert 0, "Invalid Environment"

    # initialize configurations
    env_id_for_cfg = "".join(args.env.split("-"))
    configs.update(eval("CONFIGS[{}][{}].{}config".format('"' + args.alg + '"', '"' + env_id_for_cfg + '"', args.alg)))
    configs['n_states'] = n_states
    configs['n_action_dims'] = n_action_dims
    configs['dicrete_action'] = DICRETE_ACTION_SPACE
    if n_actions:
        configs['n_actions'] = n_actions
    configs['reward_type'] = args.reward

    # for hindsight algorithms, init goal space of the environment.
    if args.alg in {"HTRPO", "HPG"}:
        configs['other_data'] = env.reset()
        assert isinstance(configs['other_data'], dict), \
            "Please check the environment settings, hindsight algorithms only support goal conditioned tasks."
        configs['reward_fn'] = env.compute_reward
        configs['max_episode_steps'] = env.max_episode_steps

    # init agent
    if args.alg in ("PG", "NPG", "TRPO", "PPO", "AdaptiveKLPPO", "HTRPO", "HPG"):
        if DICRETE_ACTION_SPACE:
            RL_brain = eval(args.alg + "_Softmax(configs)")
        else:
            RL_brain = eval(args.alg + "_Gaussian(configs)")
    else:
        RL_brain = eval(args.alg + "(configs)")

    if not args.cpu:
        RL_brain.cuda()

    # resume networks
    output_dir = os.path.join("output", "models", args.alg, env_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    RL_brain.load_model(load_path=output_dir, load_point=args.checkpoint)

    if args.usedemo:
        # TODO: demonstrate demo episodes
        pass

    run_test(env, RL_brain, num_evals = args.num_evals, render = args.render, greedy=args.greedy)
