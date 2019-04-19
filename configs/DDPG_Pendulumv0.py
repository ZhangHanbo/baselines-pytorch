from torch import optim
DDPGconfig = {
        'memory_size': 10000,
        'n_states': 3,
        'n_action_dims':1,
        'action_bounds':2,
        'batch_size':32,
        'noise_var': 3,
        'noise_min': 0.25,
        'noise_decrease': 0.0005,
        'reward_decay': 0.99,
        'lr': 0.01,
        'lr_a': 0.001,
        'tau': 0.01,
        'dicrete_action': False,
        'optimizer': optim.Adam,
    }