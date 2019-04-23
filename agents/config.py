from torch import optim
from torch.nn import MSELoss
import torch.nn.functional as F

AGENT_CONFIG = {
    # 'lr' indicates the learning rate of the "mainbody".
    # Value-based RL: Q net; Policy-based RL: Policy net; Actor-critic RL: Actor
    'lr':0.01,
    'mom':None,
    'reward_decay':0.9,
    'memory_size': 10000,
    # 'hidden_layers' defines the layers of the "mainbody".
    # Value-based RL: Q net; Policy-based RL: Policy net; Actor-critic RL: Actor
    'hidden_layers':[50],
}

DQN_CONFIG = {
    'replace_target_iter':600,
    'e_greedy':0.9,
    'e_greedy_increment':None,
    'optimizer': optim.RMSprop,
    'loss' : MSELoss,
    'batch_size': 32,
    'act_func': F.tanh,
    'out_act_func': None,
}

DDPG_CONFIG = {
    'tau' : 0.001,
    'noise_var' : 3.,
    'noise_min' : 0.,
    'noise_decrease' : 0.0005,
    'optimizer_a': optim.Adam,
    'optimizer_c': optim.Adam,
    'lr_v' : 1e-2,
    'hidden_layers_v' : None,
    'critic_loss': MSELoss,
    'batch_size': 32,
    'act_func': F.tanh,
    'out_act_func': F.tanh,
}

NAF_CONFIG = {
    'tau' : 0.001,
    'noise_var' : 3.,
    'noise_min' : 0.,
    'noise_decrease' : 0.0005,
    'optimizer': optim.Adam,
    'loss': MSELoss,
    'batch_size': 32,
    'act_func': F.tanh,
    'out_act_func': F.tanh,
}

PG_CONFIG = {
    'optimizer':optim.Adam,
    'using_batch' : False,
    'value_type' : None,
    'hidden_layers_v': None,
    'GAE_lambda' : 0.97,            # HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. 2016 ICLR
    'loss_func_v':MSELoss,
    'v_optimizer':optim.LBFGS,
    'lr_v' : 0.01,
    'entropy_weight':0.01,
    'mom_v' : None,
    'act_func': F.tanh,
    'out_act_func': None,
}

NPG_CONFIG = {
    'cg_iters': 10,
    'cg_residual_tol' : 1e-10,
    'cg_damping': 1e-3,
    'max_kl_divergence':0.01
}

PPO_CONFIG = {
    'using_batch': True,
    'steps_per_update': 10,
    'clip_epsilon': 0.2,
    'max_grad_norm': 40
}

AdaptiveKLPPO_CONFIG = {
    'using_batch':True,
    'init_beta':3.,
    'steps_per_update': 10
}

TRPO_CONFIG = {
    'max_search_num' : 10,
    'accept_ratio' : .1,
    'step_frac': .5
}
