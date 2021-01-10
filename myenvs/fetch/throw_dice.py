import os
from gym import utils as gym_utils
from . import fetch_env, utils, rotations
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('throw_dice.xml')

class FetchThrowDiceEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):

        self._dice_eulers = [
            [0., np.pi / 2, 0.],
            [0., - np.pi / 2, 0.],
            [np.pi / 2, 0., 0.],
            [-np.pi / 2, 0., 0.],
            [0., 0., 0.],
            [0., -np.pi, 0.]
        ]
        self._dice_poses = [
            rotations.euler2quat(np.array(euler)) for euler in self._dice_eulers
        ]

        self._dice_rot_mats = [
            rotations.euler2mat(np.array(euler)) for euler in self._dice_eulers
        ]

        self._dice_norms = np.array([
            np.mat(m).I.dot(np.array(((0,), (0,), (1,)))).tolist() for m in self._dice_rot_mats
        ]).squeeze()


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
            initial_qpos=initial_qpos, reward_type=reward_type, n_actions=8)
        gym_utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        return - (achieved_goal.squeeze(-1) != goal.squeeze(-1)).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        return (achieved_goal.squeeze(-1) == desired_goal.squeeze(-1)).astype(np.float32)

    def _sample_goal(self):
        # select a target pose for the dice
        goal = np.random.randint(6)
        return np.array([goal])


    def _render_callback(self):
        # Visualize target.

        ################################################################
        # model.site_pos is the relative site position to the parent body
        # data.site_xpos is the absolute site position in the world frame
        ################################################################

        object_qpos = self.sim.data.get_joint_qpos('target0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:3] = np.array([1.32441906, 0.75018422, 0.301])
        obj_quat = self._dice_poses[self.goal[0]]
        object_qpos[3:] = obj_quat
        self.sim.data.set_joint_qpos('target0:joint', object_qpos)
        self.sim.forward()


    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt

        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        in_the_air = False
        if object_pos[2] > 0.43:
            in_the_air = True

        if not in_the_air:
            vert_vec = ((0,), (0,), (1,))
            obj_rotmat = self.sim.data.get_site_xmat('object0')
            obj_inv_rotmat = np.mat(obj_rotmat).I
            vert_vec = obj_inv_rotmat.dot(np.array(vert_vec)).reshape(1, 3)
            achieved_goal = np.array([np.linalg.norm(vert_vec - self._dice_norms, axis=1).argmin()])
        else:
            achieved_goal = np.array([-1], dtype=np.int32)

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

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
            action = np.array([0., 0., 0., 1., 1., 0., 1., 0.])
        elif mode == "close":
            action = np.array([0., 0., 0., -1., 1., 0., 1., 0.])
        elif mode == "raise":
            action = np.array([0., 0., 0.5, -1., 1., 0., 1., 0.])
        else:
            raise ValueError
        for _ in range(10):
            self.step(action)




