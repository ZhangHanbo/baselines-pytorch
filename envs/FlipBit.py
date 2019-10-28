import re
import numpy as np
from scipy.sparse.csgraph import shortest_path
from gym import spaces
class FlipBit(object):
    def __init__(self, n_bits = 8):
        self.n_actions = n_bits
        self.action_space = spaces.Discrete(n_bits)
        self.d_observations = n_bits
        self.d_goals = n_bits
        self.observation_space = spaces.Dict({
            "observation": spaces.MultiBinary(n_bits),
            "desired_goal": spaces.MultiBinary(n_bits),
            "achieved_goal": spaces.MultiBinary(n_bits),
        })
        self.max_episode_steps = n_bits

    def reset(self):
        self.n_steps = 0
        self.state = np.zeros(self.d_observations)
        # self.state = np.random.randint(0, 2, size=self.d_observations)
        self.goal = np.random.randint(0, 2, size=self.d_goals)
        state, goal = np.array(self.state), np.array(self.goal)
        obs = {
            "observation": state,
            "desired_goal": goal,
            "achieved_goal": state.copy(),
        }
        return obs

    def step(self, a):
        if a[0] >= self.n_actions:
            raise Exception('Invalid action')
        self.n_steps += 1
        self.state[a[0]] = 1 - self.state[a[0]]
        if np.allclose(self.state, self.goal):
            reward = 0.
        else:
            reward = -1.
        done = (self.max_episode_steps <= self.n_steps) or (reward == 0.)

        obs = {
            "observation": self.state,
            "desired_goal": self.goal,
            "achieved_goal": self.state.copy(),
        }

        info = {'is_success': reward == 0}
        if done:
            info['episode'] = {
                'l' : self.n_steps,
                'r' : - self.n_steps + 1 + reward,
            }

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal.shape == desired_goal.shape
        dif = np.abs((achieved_goal - desired_goal)).sum(axis=-1)
        return - (dif > 0).astype(np.float32)

    def render(self):
        print(self.__repr__())

    def seed(self, seed):
        np.random.seed(seed)

    def __repr__(self):
        return 'State: {0}. Goal: {1}.'.format(self.state, self.goal)

class FlipBit8(FlipBit):
    def __init__(self):
        super(FlipBit8, self).__init__(n_bits = 8)

class FlipBit16(FlipBit):
    def __init__(self):
        super(FlipBit16, self).__init__(n_bits = 16)
