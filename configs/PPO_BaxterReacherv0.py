from torch import optim
import torch.nn.functional as F
# command line args: --seed 4 --algo PPO --env Reacher-v1 --num_steps 500 --num_episodes 10000

PPOconfig = {
        'n_states': 30,
        'n_action_dims': 7,
        'action_bounds': 1,
        'memory_size': 1000,
        'reward_decay': 0.95,
        'steps_per_update': 64,
        'batch_size': 256,
        'max_grad_norm': 2,
        'GAE_lambda': 0.95,
        'clip_epsilon': 0.2,
        'lr': 1e-3,
        'lr_v': 1e-3,
        'hidden_layers' : [128,128],
        'hidden_layers_v' : [128,128],
        'entropy_weight' : 0,
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC',
        'dicrete_action': False,
        'act_func':F.tanh,
        'out_act_func': F.tanh,
        'num_episode':10000,
        'num_step':50,
    }