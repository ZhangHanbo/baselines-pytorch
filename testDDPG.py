import gym
import torch
import numpy as np
from agent.DDPG import DDPG

def run_game(env, agent):
    step = 0
    RENDER = False
    for i_episode in range(200):
        observation = env.reset()
        total_r = 0
        for t in range(200):
            if RENDER:
                env.render()
            # choose action
            action = agent.choose_action(observation)
            action = np.clip(action, -2, 2)
            # transition
            observation_, reward, done, info = env.step(action)
            total_r = total_r + reward
            # store transition
            agent.store_transition(observation, action, reward/10, observation_, done)
            # update parameters
            if step > agent.memory_size:
                agent.learn()

            # swap observation
            observation = observation_

            step = step + 1

            if done:
                #print('reward: ' + str(total_r) + ' episode: ' + str(i_episode) + ' explore: ' + str(agent.noise))
                #print("Episode finished after {} timesteps".format(t + 1))
                break

        if total_r > -300:
            RENDER = True
        print('reward: ' + str(total_r) + ' episode: ' + str(i_episode) + ' explore: ' + str(agent.noise))



if __name__ == "__main__":
    # maze game
    env = gym.make('Pendulum-v0')
    #env = env.unwrapped
    #env.seed(1)
    config = {
        'n_features' : env.observation_space.shape[0],
        'n_actions' : env.action_space.shape[0],
        'action_bounds' : env.action_space.high,
        'noise_var': 3,
        'noise_min':0.05,
        'noise_decrease':0.0005,
        'lr':0.002,
        'lr_a':0.001,
        'tau':0.01
    }

    RL = DDPG(config)
    run_game(env,RL)

