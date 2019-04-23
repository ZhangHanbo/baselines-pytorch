from torch import optim
DDPGconfig = {
        'memory_size': 100000,
        'n_states': 11,
        'n_action_dims':2,
        'action_bounds':1,
        'batch_size':4096,
        'noise_var': 1.,
        'noise_min': 0.,
        'noise_decrease': 0.0001,
        'reward_decay': 0.95,
        'lr': 0.001,
        'lr_v': 0.001,
        'tau': 0.001,
        'dicrete_action': False,
        'hidden_layers' : [128,128],
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
    }