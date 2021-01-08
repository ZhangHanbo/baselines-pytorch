from gym.core import Wrapper
import time

# TODO: implement the sparse reward wrapper here
class FinalAccSparse(Wrapper):

    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        assert hasattr(env, "max_episode_steps"), "The environment should be time limited"
        self.max_episode_steps = env.max_episode_steps

    def step(self, action):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        pass