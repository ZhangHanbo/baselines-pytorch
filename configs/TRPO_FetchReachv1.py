from torch.nn import functional as F
from torch import optim

TRPOconfig = {
    'cg_damping': 1e-1,
    'cg_residual_tol' : 1e-10,
    'reward_decay': 0.98,
    'GAE_lambda': 0.,
    'max_kl_divergence': 0.01,
    'entropy_weight': 0,
    'hidden_layers': [256, 256, 256],
    'hidden_layers_v': [256, 256, 256],
    'max_grad_norm': None,
    'lr_v': 5e-4,
    'iters_v':20,
    'v_optimizer': optim.Adam,
    'steps_per_iter': 1000,
    'max_search_num': 10,
    'accept_ratio': .1,
    'step_frac': .5,
    'using_KL_estimation': False,
}
TRPOconfig['memory_size'] = TRPOconfig['steps_per_iter']
