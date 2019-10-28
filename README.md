# Deep Reinforcement Learning Package

## Introduction

This is a deep reinforcement learning package including main-stream RL algorithms such as DDPG, TRPO, PPO, etc. It will be updated continuously to contain up-to-date main-stream RL algorithms. It aims to help all the RL newers and researchers more easily understand and utilize RL algorithms in their own research. All the included algorithms will be implemented so that they can achieve the claimed performance in the corresponding papers. By now, the following algorithms are nearly-SOTA:

**DDPG, NAF, TD3, TRPO, PPO** 

<font size=2 color=grey>**The Definition of "nearly-SOTA"**: *I don't have enough time to test all the envs included in the corresponding papers and provide the comparison with the baselines. I just test my implemented version in one of the envs (most probable the Hopper-v2) and it achieved the same or higher performance compared with the baseline.*</font>

It also includes our newly proposed algorithm **Hindsight Trust Region Policy Optimization**. A demo video of **Hindsight Trust Region Policy Optimization** is in demo/ which shows how our algorithm works. **Hindsight Trust Region Policy Optimization** has already been submitted to ICLR 2020.

### Requirements

python               3.4

torch                1.1.0

numpy                1.16.2

gym                  0.12.1

tensorboardX         1.7

mujoco-py            2.0.2.2

Please make sure that the versions of all the requirements match the ones above, which is necessary for running the code.

## Implemented Algorithms

|Alg. | SOTA?|
|  ----  | ----  |
Deep Q-Learning (DQN) | × 
Double DQN | × 
Dueling DQN| × 
Normalized Advantage Function with DQN | √
Deep  Deterministic Policy Gradient (DDPG) | √
Twin Delayed DDPG | √
Vanilla Policy Gradient| -
Natual Policy Gradient | -
Trust Region Policy Optimization | √
Proximal Policy Optimization | √
Hindsight Trust Region Policy Gradient | √

### Examples
For running continuous envs (e.g. FetchPush-v1) with HTRPO algorithm:
```bash
python main.py --alg HTRPO --env FetchPush-v1 --num_steps 2000000 --num_evals 200 --eval_interval 19200 (--cpu)
```

For running discrete envs (e.g. FlipBit8):

```bash
python main.py --alg HTRPO --env FlipBit8 --unnormobs --num_steps 50000 --num_evals 200 --eval_interval 1024 (--cpu)
```

--cpu is used only when you want to train the policy using CPU, which will be much slower than using GPU.

--unnormobs is used when you do not want to do input normalization. In our paper, all the discrete envs do not use this trick at all.

**Note** for users: 

1. DDPG, TD3 and NAF should open the switches named "unnormobs" and "unnormret" during training. The normalization is not optimized for these 3 methods by now and with observation normalization or return normalization, the performance will be much lower than the baselines.

2. We propose HTRPO for sparse reward reinforcement learning, and as baselines, TRPO and HPG for sparse reward are also implemented. To run HTRPO, you need to follow the above instruction. To run HPG, you only need to modify the hyperparameter "using_hpg" to "True" in the corresponding config file (e.g. for FetchPush-v1, the config file is configs/HTRPO_FetchPushv1.py). To run HTRPO, you need to modify the hyperparameter "sampled_goal_num" to 0 and "using_original_data" to "True", which means that the policy will be trained using only the original collected data without any modification.

3. All the experimental results compared with baselines will be continuous updated when I have time.

### Environment List

1. All the mojoco envs.

2. Our Discrete Envs: FlipBit8, FlipBit16, EmptyMaze, FourRoomMaze, FetchReachDiscrete, FetchPushDiscrete

All the listed names can be directly used in command line for training policies. BUT NOTE: sparse reward envs only support HTRPO and dense reward envs do not support HTRPO.

### Papers
1. Zhang, Hanbo, et al. "Hindsight Trust Region Policy Optimization." arXiv preprint arXiv:1907.12439 (2019).
2. Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
3. Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.
4. Wang, Ziyu, et al. "Dueling Network Architectures for Deep Reinforcement Learning." International Conference on Machine Learning. 2016.
3. Gu, Shixiang, et al. "Continuous deep q-learning with model-based acceleration." International Conference on Machine Learning. 2016.
4. Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).
5. Fujimoto, Scott, Herke Hoof, and David Meger. "Addressing Function Approximation Error in Actor-Critic Methods." International Conference on Machine Learning. 2018.
6. Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." Advances in neural information processing systems. 2000.
7. Kakade, Sham M. "A natural policy gradient." Advances in neural information processing systems. 2002.
8. Schulman, John, et al. "Trust region policy optimization." International conference on machine learning. 2015.
9. Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
10. Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).