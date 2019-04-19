from torch import optim

PGconfig = {
        'n_states': 3,
        'n_action_dims': 1,
        'lr': 3e-3,
        'memory_size': 500,
        'reward_decay': 0.995,
        'batch_size': 500,
        'GAE_lambda': 0.97,
        'value_type' : 'FC',
        'optimizer': optim.Adam
    }