import gym
import torch
import numpy as np
import agents
from torch import optim

# if true, the done of the final step in each episode will be set to True.
FINALDONE =True

def run_game(env, agent):
    step = 0
    RENDER = False
    for i_episode in range(3000):
        observation = env.reset()
        total_r = 0
        episode_lenth = 200
        for t in range(episode_lenth):
            if RENDER:
                env.render()
            # choose action
            action, mu ,sigma = agent.choose_action(observation)
            action = np.clip(action, -2, 2)
            # transition
            observation_, reward, done, info = env.step(action)
            if t == episode_lenth - 1:
                done = True
            total_r = total_r + reward
            # store transition
            transition = {
                'state': np.expand_dims(observation, 0),
                'action': np.expand_dims(action, 0),
                'mu': np.expand_dims(mu, 0),
                'sigma': np.expand_dims(sigma, 0),
                'reward': np.expand_dims(np.array([reward / 10]), 0),
                'next_state': np.expand_dims(observation_, 0),
                'done':np.expand_dims(np.array([done]), 0),
            }
            agent.store_transition(transition)
            # swap observation
            observation = observation_
            step = step + 1

        print('episode: ' + str(i_episode) + '   reward: ' + str(total_r))
        print("Episode finished after {} timesteps".format(t + 1))

        if (i_episode +1) % (agent.memory_size / episode_lenth) ==0:
            agent.learn()

        if total_r > -20:
            RENDER = True

def run_game_to_end(env, agent):
    step = 0
    RENDER = False
    for i_episode in range(3000):
        memory_count = 0
        while(True):
            total_r = 0
            observation = env.reset()
            while (True):
                if RENDER:
                    env.render()
                # choose action
                action, mu, sigma = agent.choose_action(observation)
                action = np.clip(action, -2, 2)
                # transition
                observation_, reward, done, info = env.step(action)
                total_r = total_r + reward
                # store transition
                transition = torch.Tensor(np.hstack((observation, mu, sigma, action, reward / 10, done, observation_)))
                agent.store_transition(transition)
                memory_count += 1
                # swap observation
                observation = observation_
                step = step + 1
                if done:
                    # print('reward: ' + str(total_r) + ' episode: ' + str(i_episode) + ' explore: ' + str(agent.noise))
                    # print("Episode finished after {} timesteps".format(t + 1))
                    break
            print('episode: ' + str(i_episode) + '   reward: ' + str(total_r))
            if memory_count> agent.memory_size:
                break
        agent.learn()
        if total_r>70:
            RENDER = True



if __name__ == "__main__":
    # maze game
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    # env.seed(1)
    PPO_config = {
        'n_states': env.observation_space.shape[0],
        'n_action_dims': env.action_space.shape[0],
        'action_bounds': env.action_space.high,
        'memory_size':600,
        'reward_decay':0.95,
        'steps_per_update':15,
        'batch_size':3000,
        'max_grad_norm': 2,
        'GAE_lambda':0.95,
        'clip_epsilon': 0.2,
        'lr' : 3e-2,
        'lr_v':3e-2,
        'optimizer': optim.Adam,
        'v_optimizer': optim.Adam,
        'value_type': 'FC'
    }

    TRPO_config = {
        'n_states': env.observation_space.shape[0],
        'n_action_dims': env.action_space.shape[0],
        'action_bounds': env.action_space.high,
        'memory_size': 600,
        'reward_decay': 0.95,
        'GAE_lambda': 1,
        'lr_v': 3e-2,
        'v_optimizer': optim.LBFGS,
        'value_type': 'FC'
    }

    DDPG_config = {
        'noise_var': 3,
        'noise_min': 0.05,
        'noise_decrease': 0.0005,
        'lr': 0.001,
        'lr_a': 0.001,
        'tau': 0.01
    }

    RL = agents.TRPO_Gaussian(TRPO_config)
    run_game(env,RL)

