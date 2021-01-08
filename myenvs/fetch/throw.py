import os
from gym import utils as gym_utils
from . import fetch_env, utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('throw.xml')


class FetchThrowEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.15, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        gym_utils.EzPickle.__init__(self)


    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self._adjust_gripper(mode="open")

        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2]
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self._adjust_gripper(mode="close")
        self._adjust_gripper(mode="raise")

        self.sim.forward()

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

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        init_gripper_target = \
            np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = init_gripper_target.copy()
        # randomize the initial localization of the gripper and object
        while np.linalg.norm(gripper_target[:2] - init_gripper_target[:2]) < 0.1:
            gripper_target[:2] = init_gripper_target[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]