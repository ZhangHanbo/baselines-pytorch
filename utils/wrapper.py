import gym
import numpy as np
import time

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class CumSparseReward(gym.Wrapper):

    def __init__(self, env):
        super(CumSparseReward, self).__init__(env)
        assert hasattr(env, "max_episode_steps"), "The environment should be time limited"
        self.n_step = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        assert rew == -1 or rew == 0, "CumSparseReward wrapper only supports sparse reward envs."
        rew = 0 if rew == -1 else self.max_episode_steps - self.n_step
        done = True if rew > 0 else False
        self.n_step += 1
        return obs, rew, done, info

    def reset(self, action):
        self.n_step = 0
        self.env.reset()

    def compute_reward(self, achieved_goal, desired_goal, info):
        # info should include the timestep infomation
        curr_steps = info["steps"]

class ActionNormalizer(gym.Wrapper):
    def __init__(self, env):
        super(ActionNormalizer, self).__init__(env)
        self.ori_action_space = env.action_space
        assert (self.ori_action_space.low > - np.inf).sum() == self.ori_action_space.low.size
        assert (self.ori_action_space.high < np.inf).sum() == self.ori_action_space.high.size
        self.ori_low = self.ori_action_space.low
        self.ori_high = self.ori_action_space.high

    def step(self, a):
        a = (a + 1.) / 2 * (self.ori_high - self.ori_low) + self.ori_low
        return self.env.step(a)
