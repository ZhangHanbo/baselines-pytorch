from torch import optim
NAFconfig = {
        'memory_size': 2000,
        'n_states': 3,
        'n_action_dims':1,
        'action_bounds':2,
        'batch_size':32,
        'noise_var': 3,
        'noise_min': 0.3,
        'noise_decrease': 0.0005,
        'reward_decay': 0.95,
        # Higher learning rate will cause faster convergence but be unstable. [0.001, 0.01] is recommanded.
        'lr': 0.001,
        'tau': 0.001,
        'dicrete_action': False,
        'hidden_layers' : [50],
        'use_batch_norm' : False,
        'optimizer': optim.Adam,
    }