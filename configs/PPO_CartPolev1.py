from torch import optim
import torch.nn.functional as F

PPOconfig = {
        'n_states': 4,
        'n_action_dims': 1,
        'n_actions':2,
        'memory_size':3000,
        'reward_decay':0.95,
        'steps_per_update':15,
        'batch_size':256,
        'max_grad_norm': 2,
        'GAE_lambda':0.95,
        'clip_epsilon': 0.2,
        'lr' : 3e-2,
        'lr_v':3e-2,
        'hidden_layers' : [50],
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC',
        'dicrete_action': True,
        'out_act_func': F.softmax,
        'num_episode': 10000,
    }