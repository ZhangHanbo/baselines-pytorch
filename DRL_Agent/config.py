from torch import optim
from torch.nn import MSELoss

AGENT_CONFIG = {
    'lr':0.01,
    'mom':None,
    'reward_decay':0.9,
    'batch_size':32,
    'memory_size': 10000,
}

DQN_CONFIG = {
    'replace_target_iter':600,
    'e_greedy':0.9,
    'e_greedy_increment':None,
    'optimizer': optim.RMSprop,
    'loss' : MSELoss
}

DDPG_CONFIG = {
    'tau' : 0.001,
    'noise_var' : 3.,
    'noise_min' : 0,
    'noise_decrease' : 0.0005,
    'optimizer_a': optim.Adam,
    'optimizer_c': optim.Adam,
    'lr_a' : 1e-3,
    'critic_loss': MSELoss
}

NAF_CONFIG = {
    'tau' : 0.001,
    'noise_var' : 3.,
    'noise_min' : 0,
    'noise_decrease' : 0.0005,
    'optimizer': optim.Adam,
    'loss': MSELoss
}

PG_CONFIG = {
    'optimizer':optim.Adam,
    'value_type' : 'FC',
    'GAE_lambda' : 0.97,            # HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. 2016 ICLR
    'loss_func_v':MSELoss,
    'v_optimizer':optim.LBFGS
}

NPG_CONFIG = {
    'cg_iters': 10,
    'cg_residual_tol' : 1e-10,
    'cg_damping': 1e-3,
    'max_kl_divergence':0.01
}

PPO_CONFIG = {
    'steps_per_update': 10,
    'clip_epsilon': 0.2
}

AdaptiveKLPPO_CONFIG = {
    'init_beta':3.,
    'steps_per_update': 10
}

TRPO_CONFIG = {
    'max_search_num' : 10,
    'accept_ratio' : .1,
    'step_frac': .5
}
