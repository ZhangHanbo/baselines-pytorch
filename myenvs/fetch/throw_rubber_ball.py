import os
from gym import utils as gym_utils
from . import fetch_env, utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('throw_rubber_ball.xml')


class FetchThrowRubberBallEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.15, target_range=0.3, distance_threshold=0.05,
            initial_qpos=self.initial_qpos, reward_type=reward_type)
        gym_utils.EzPickle.__init__(self)


    def _reset_sim(self):
        # self._env_setup(initial_qpos=self.initial_qpos)

        self.sim.set_state(self.initial_state)

        # after setting states, the simulator should forward for one time to update the state
        self.sim.forward()

        self._adjust_gripper(mode="open")

        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2]
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()

        self._adjust_gripper(mode="close")
        self._adjust_gripper(mode="raise")

        return True


    def _adjust_gripper(self, mode="open"):
        if mode == "open":
            action = np.array([0., 0., 0., 1.])
        elif mode == "close":
            action = np.array([0., 0., 0., -1.])
        elif mode == "raise":
            action = np.array([0., 0., 0.5, -1.])
        else:
            raise ValueError

        for _ in range(10):
            self.step(action)

        if mode == "raise":
            for _ in range(10):
                action = (np.random.rand(4) - 0.5) * 2
                action[-1] = -1
                self.step(action)
