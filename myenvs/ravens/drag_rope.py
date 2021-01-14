import pybullet as p
import numpy as np
import os

from ravens.environments.environments import *
from ravens.utils import pybullet_utils
from ravens.tasks.manipulating_rope import ManipulatingRope

class DragRope(Environment):

    def __init__(self):
        super(DragRope, self).__init__(assets_root="ravensenvs/", disp=False,
                                             shared_memory=False, hz=480)
        task = ManipulatingRope()
        self.set_task(task)


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


    def step(self, action=None):
        """Execute action with specified primitive.
            Args:
              action: action to execute.
            Returns:
              (obs, reward, done, info) tuple containing MDP step data.
            """
        if action is not None:
            timeout = self.task.primitive(self.movej, self.movep, self.ee, **action)

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = {'color': (), 'depth': ()}
                for config in self.agent_cams:
                    color, depth, _ = self.render_camera(config)
                    obs['color'] += (color,)
                    obs['depth'] += (depth,)
                return obs, 0.0, True, self.info

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            p.stepSimulation()

        # Get task rewards.
        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()

        # Add ground truth robot state into info.
        info.update(self.info)

        # Get RGB-D camera image observations.
        obs = {'color': (), 'depth': ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)

        return obs, reward, done, info

