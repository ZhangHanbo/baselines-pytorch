from torch import optim

TRPOconfig = {
        'n_states': 3,
        'n_action_dims': 1,
        'action_bounds': 2,
        'memory_size': 3000,
        'reward_decay': 0.95,
        'GAE_lambda': 0.95,
        'lr_v': 1e-2,
        'v_optimizer': optim.LBFGS,
        'value_type': 'FC',
        'dicrete_action': False
    }