import gym
import torch
import numpy as np
from DRL_Agent import DDPG, NAF, Agent, TRPO_Gaussian, PG_Gaussian, NPG_Gaussian

# if true, the done of the final step in each episode will be set to True.
FINALDONE =True

def run_game(env, agent):
    step = 0
    RENDER = False
    for i_episode in range(6000):
        observation = env.reset()
        total_r = 0
        episode_lenth = 400
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
            transition = torch.Tensor(np.hstack((observation, mu,sigma ,action, reward/10, done, observation_)))
            agent.store_transition(transition)
            # swap observation
            observation = observation_
            step = step + 1
            if done:
                #print('reward: ' + str(total_r) + ' episode: ' + str(i_episode) + ' explore: ' + str(agent.noise))
                #print("Episode finished after {} timesteps".format(t + 1))
                break
        agent.learn()
        print('episode: '+ str(i_episode)+'   reward: '+str(total_r))

        if total_r > -300:
            RENDER = True



if __name__ == "__main__":
    # maze game
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    config = {
        'n_features': env.observation_space.shape[0],
        'n_actions': env.action_space.shape[0],
        'action_bounds': env.action_space.high,
        'memory_size':2000,
        'reward_decay':0.9,
        'lr' : 0.01
    }
    '''
    'noise_var': 3,
    'noise_min':0.05,
    'noise_decrease':0.0005,
    'lr':0.001,
    'lr_a': 0.001,
    'tau':0.01
    '''

    RL = TRPO_Gaussian(config)
    run_game(env,RL)

