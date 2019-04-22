from torch import optim

TRPOconfig = {
        'n_states': 4,
        'n_action_dims': 1,
        'n_actions': 2,
        'memory_size': 3000,
        'reward_decay': 0.95,
        'GAE_lambda': 1,
        'lr_v': 3e-2,
        'hidden_layers' : [50],
        'use_batch_norm' : False,
        'v_optimizer': optim.LBFGS,
        'value_type': 'FC',
        'dicrete_action': True,
    }