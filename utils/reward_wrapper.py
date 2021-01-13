from gym.core import Wrapper
import time

class CumSparseReward(Wrapper):

    def __init__(self, env):
        Wrapper.__init__(self, env=env)
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

