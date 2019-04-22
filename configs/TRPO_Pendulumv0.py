from torch import optim

TRPOconfig = {
        'n_states': 3,
        'n_action_dims': 1,
        'action_bounds': 2,
        'memory_size': 2000,
        'reward_decay': 0.95,
        'GAE_lambda': 0.95,
        'hidden_layers' : [50],
        'use_batch_norm' : False,
        'lr_v': 0.1,
        'v_optimizer': optim.LBFGS,
        'value_type': 'FC',
        'dicrete_action': False
    }