from torch import optim

DQNconfig = {
        'n_states':4,
        'dicrete_action': True,
        'n_actions':2,
        'n_action_dims': 1,
        'lr':0.01,
        'mom':0,
        'reward_decay':0.9,
        'e_greedy':0.9,
        'replace_target_iter':100,
        'memory_size': 2000,
        'batch_size': 32,
        'e_greedy_increment':None,
        'optimizer': optim.Adam
    }