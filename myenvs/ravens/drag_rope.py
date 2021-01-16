import pybullet as p
import numpy as np
import os
import gym
from gym import spaces

from ravens.environments.environments import *
from ravens.utils import pybullet_utils, utils
from ravens.tasks.manipulating_rope import ManipulatingRope


class Drag():
    """Pick and place primitive."""

    def __init__(self, movep, movej, ee, height=0.32, speed=0.01):
        self.height, self.speed = height, speed
        self.movep, self.movej, self.ee = movep, movej, ee
        self.picked = False

    def init_pick(self, pick_pose):
        # Execute picking primitive.
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
        prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
        postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
        timeout = self.movep(prepick_pose)

        # Move towards pick pose until contact is detected.
        delta = (np.float32([0, 0, -0.001]),
               utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose
        while not self.ee.detect_contact():  # and target_pose[2] > 0:
            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= self.movep(targ_pose)
            if timeout:
                return True

        # Activate end effector, move up, and check picking success.
        self.ee.activate()
        timeout |= self.movep(postpick_pose, self.speed)
        pick_success = self.ee.check_grasp()
        return pick_success

    def __call__(self, place_pose):
        delta = (np.float32([0, 0, -0.001]),
                 utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        self.picked = self.ee.check_grasp()

        # Execute placing primitive if pick is successful.
        timeout = False
        if self.picked:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            preplace_pose = utils.multiply(place_pose, preplace_to_place)
            targ_pose = preplace_pose
            while not self.ee.detect_contact():
                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= self.movep(targ_pose, self.speed)
                if timeout:
                    return True
        return timeout


class DragRopeEnv(Environment):

    def __init__(self):
        super(DragRopeEnv, self).__init__(assets_root="ravensenvs/", disp=False,
                                             shared_memory=False, hz=480)
        task = ManipulatingRope()
        task.primitive = Drag(self.movep, self.movej, self.ee)
        self.set_task(task)
        self.task.reset()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.d_obs = self._get_obs().shape
        self.observation_space = spaces.Dict({
            "observation":spaces.Box(-np.inf, np.inf, shape=self.d_obs, dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=6, dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=6, dtype=np.float32),
        })
        self.dist_thresh = 0.03


    def _get_obs(self):
        obj_states = []
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                obj_states.append(np.concatenate(p.getBasePositionAndOrientation(obj_id)))
        obj_states = np.concatenate(obj_states)
        return obj_states


    def _get_achieved_goal(self):
        rope_particle_ids = (self.task.goals[0][0][0], self.task.goals[0][-1][0])
        end_particle_pos = (p.getBasePositionAndOrientation(i)[0] for i in rope_particle_ids)
        return np.concatenate(end_particle_pos)


    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError('environment task must be set. Call set_task or pass '
                             'the task arg in the environment constructor.')
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                                 [0, 0, -0.001])
        pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])

        # Load UR5 robot arm equipped with suction end effector.
        # TODO(andyzeng): add back parallel-jaw grippers.
        self.ur5 = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, UR5_URDF_PATH))
        self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
        self.ee_tip = 10  # Link ID of suction cup.

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Reset end effector.
        self.ee.release()

        # Reset task.
        self.task.reset(self)
        self.goal = self.task.goals[-1][2]
        self.goal = np.concatenate((self.goal[0][0], self.goal[-1][0]))

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        observation = self._get_obs()
        achieved_goal = self._get_achieved_goal()
        obs = {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

        return obs


    def compute_reward(self, achieved_goal, desired_goal, info):
        _shape = achieved_goal.shape[:-1] + (len(self.key_point_indices), 3)
        achieved_goal = achieved_goal.reshape(_shape)
        desired_goal = desired_goal.reshape(_shape)

        if self._rope_symmetry:
            # the symmetry of the rope
            dist1 = np.linalg.norm(achieved_goal - desired_goal, axis=-1).max(-1)
            dist2 = np.linalg.norm(np.flip(achieved_goal, axis=-2) - desired_goal, axis=-1).max(-1)
            dist = np.minimum(dist1, dist2)
        else:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1).max(-1)

        if self.reward_type == "sparse":
            return - (dist >= self.dist_thresh).astype(np.float32)
        else:
            return - dist


    def step(self, action=None):

        if action is not None:
            timeout = self.task.primitive(action)

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = self._get_obs()
                achieved_goal = self._get_achieved_goal()
                desired_goal = self.goal
                reward = self.compute_reward(achieved_goal, desired_goal, None)
                obs = {
                    "observation": obs.copy(),
                    "achieved_goal": achieved_goal.copy(),
                    "desired_goal": desired_goal.copy()
                }
                return obs, reward, True, self.info

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            p.stepSimulation()

        obs = self._get_obs()
        achieved_goal = self._get_achieved_goal()
        desired_goal = self.goal

        # Get task rewards.
        reward = self.compute_reward(achieved_goal, desired_goal, None)
        done = (reward >= - self.dist_thresh)

        info = {
            "is_success": reward >= - self.dist_thresh
        }

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal.copy()
        }

        return obs, reward, done, info

