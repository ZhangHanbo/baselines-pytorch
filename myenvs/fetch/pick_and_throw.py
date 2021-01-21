import os
from gym import utils as gym_utils
from . import fetch_env, utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('pick_and_throw.xml')


class FetchPickAndThrowEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.15, target_in_the_air=True, target_offset=np.array([0.2, 0.0, 0.0]),
            obj_range=0.15, target_range=0.3, distance_threshold=0.05,
            initial_qpos=self.initial_qpos, reward_type=reward_type)
        gym_utils.EzPickle.__init__(self)


    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3].copy()
            goal[0] += self.np_random.uniform(0, self.target_range, size=1)
            goal[1] += self.np_random.uniform(-self.target_range, self.target_range, size=1)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.15)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()
