from torch import optim

DQN_CONFIG = {
    'lr':0.01,
    'mom':0.9,
    'reward_decay':0.9,
    'e_greedy':0.9,
    'replace_target_iter':300,
    'memory_size':1000,
    'batch_size':32,
    'e_greedy_increment':None,
    'optimizer': optim.RMSprop
}

POLICY_BASED_AGENT = {

}