from torch import optim
NAFconfig = {
        'memory_size': 2000,
        'n_states': 3,
        'n_action_dims':1,
        'action_bounds':2,
        'batch_size':128,
        'noise_var': 3,
        'noise_min': 0.25,
        'noise_decrease': 0.0005,
        'reward_decay': 0.95,
        'lr': 0.001,
        'tau': 0.01,
        'dicrete_action': False,
        'optimizer': optim.Adam,
    }