HTRPOconfig = {
    'reward_decay': 0.99,
    'cg_damping': 1e-3,
    'GAE_lambda': 0.,
    'max_kl_divergence': 0.000003,
    'per_decision': True,
    'weighted_is': True,
    'using_active_goals' : True,
    'hidden_layers': [256, 256, 256],
    'hidden_layers_v': [256, 256, 256],
    'max_grad_norm': None,
    'lr_v': 1e-4,
    'iters_v':10,
    # for comparison with HPG
    'lr': 3e-4,
    # NEED TO FOCUS ON THESE PARAMETERS
    'using_hpg': False,
    'steps_per_iter': 800,
    'sampled_goal_num': 30,
    'value_type': None,
    'using_original_data': False,
    'using_kl2':True
}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']
