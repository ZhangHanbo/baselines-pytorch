from torch import optim
import torch.nn.functional as F
PGconfig = {
        'n_states': 3,
        'n_action_dims': 1,
        'lr': 3e-3,
        'memory_size': 500,
        'reward_decay': 0.995,
        'batch_size': 500,
        'GAE_lambda': 0.97,
        'value_type' : 'FC',
        'hidden_layers' : [50],
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
        'out_act_func': F.tanh,
    }