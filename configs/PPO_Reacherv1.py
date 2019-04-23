from torch import optim
import torch.nn.functional as F
# command line args: --seed 4 --algo PPO --env Reacher-v1 --num_steps 500 --num_episodes 10000

PPOconfig = {
        'n_states': 11,
        'n_action_dims': 2,
        'action_bounds': 1,
        'memory_size': 16000,
        'reward_decay': 0.95,
        'steps_per_update': 15,
        'batch_size': 4096,
        'max_grad_norm': 2,
        'GAE_lambda': 0.95,
        'clip_epsilon': 0.2,
        'lr': 1e-3,
        'lr_v': 1e-3,
        'hidden_layers' : [64,64],
        'hidden_layers_v' : [64,64],
        'entropy_weight' : 0,
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC',
        'dicrete_action': False,
        'act_func':F.tanh,
        'out_act_func': F.tanh,
    }