import pybullet as p
import numpy as np
import os
import gym
from gym import spaces

from ravens.environments.environment import *
from ravens.tasks.task import Task
from ravens.utils import pybullet_utils, utils
from ravens.tasks.grippers import Spatula


class SweepingPiles(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.ee = Spatula

    @property
    def ZONE_SIZE(self):
        return 0.12

    def reset(self, env):
        super().reset(env)

        # Add goal zone.
        zone_size = (self.ZONE_SIZE, self.ZONE_SIZE, 0)
        zone_pose = self.get_random_pose(env, zone_size)[0]
        zone_pose = (zone_pose, (0., 0., 0., 1.))
        env.add_object('zone/zone.urdf', zone_pose, 'fixed')

        # Add pile of small blocks.
        obj_pts = {}
        obj_ids = []
        for _ in range(20):
            rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.12
            ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.12
            xyz = (rx, ry, 0.01)
            theta = np.random.rand() * 2 * np.pi
            xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
            obj_pts[obj_id] = self.get_object_points(obj_id)
            obj_ids.append((obj_id, (0, None)))

        # Goal: all small blocks must be in zone.
        # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
        # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
        # self.goals.append((goal, metric))
        self.goals.append((obj_ids, np.ones((50, 1)), [zone_pose], True, False,
                           'zone', (obj_pts, [(zone_pose, zone_size)]), 1))


class SweepPileEnv(Environment):

    def __init__(self, reward_type="sparse"):
        super(SweepPileEnv, self).__init__(assets_root="ravensenvs/ravens/environments/assets", disp=False,
                                             shared_memory=False, hz=480)
        task = SweepingPiles()
        self.set_task(task)

        self.reward_type = reward_type
        self._step_wait_until_settled = False
        self._verbal_frames = True
        self._verbal_frame_interval = 50
        self.frames = []

        obs = self.reset()
        self.action_space = gym.spaces.Box(
            np.array([self.task.bounds[0][0] + 0.1, self.task.bounds[1][0] + 0.1, -np.pi]),
            np.array([self.task.bounds[0][1] - 0.1, self.task.bounds[1][1] - 0.1, np.pi]),
            shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation":spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
        })

    def push_from_to(self, pose0, pose1):
        """Execute pushing primitive.

            Args:
                movej: function to move robot joints.
                movep: function to move robot end effector pose.
                ee: robot end effector.
                pose0: SE(3) starting pose.
                pose1: SE(3) ending pose.

            Returns:
                timeout: robot movement timed out if True.
        """
        # Adjust push start and end positions.
        pos0 = np.float32((pose0[0], pose0[1], 0.005))
        pos1 = np.float32((pose1[0], pose1[1], 0.005))
        vec = np.float32(pos1) - np.float32(pos0)
        length = np.linalg.norm(vec)
        vec = vec / length
        pos0 -= vec * 0.02
        pos1 -= vec * 0.05

        # Align spatula against push direction.
        theta = np.arctan2(vec[1], vec[0])
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        over0 = (pos0[0], pos0[1], 0.31)
        over1 = (pos1[0], pos1[1], 0.31)

        # Execute push.
        timeout = self.movep((over0, rot))
        timeout |= self.movep((pos0, rot))
        n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
        for _ in range(n_push):
            target = pos0 + vec * n_push * 0.01
            timeout |= self.movep((target, rot), speed=0.003)
        timeout |= self.movep((pos1, rot), speed=0.003)
        timeout |= self.movep((over1, rot))
        return timeout


    def _get_obs(self):
        obj_states = []
        for obj_id in self.obj_ids["rigid"]:
            obj_states.append(p.getBasePositionAndOrientation(obj_id)[0])
        obj_states = np.concatenate(obj_states)
        return obj_states


    def _get_achieved_goal(self, obs):
        assert len(obs.shape) in {1, 2}
        block_pos = obs.reshape(-1, 3)
        min_x = block_pos[:, 0].min()
        min_y = block_pos[:, 1].min()
        max_x = block_pos[:, 0].max()
        max_y = block_pos[:, 1].max()
        return np.asarray((min_x, min_y, max_x, max_y))


    def reset(self):
        """Performs common reset functionality for all supported tasks."""

        self.frames = []

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
        goal_pos = self.task.goals[-1][2][0][0]
        goal_size = self.task.ZONE_SIZE
        goal_x_min = goal_pos[0] - 0.5 * goal_size
        goal_y_min = goal_pos[1] - 0.5 * goal_size
        goal_x_max = goal_pos[0] + 0.5 * goal_size
        goal_y_max = goal_pos[1] + 0.5 * goal_size
        self.goal = np.asarray((goal_x_min, goal_y_min, goal_x_max, goal_y_max))

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        observation = self._get_obs()
        achieved_goal = self._get_achieved_goal(observation)
        obs = {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

        return obs


    def _get_generalized_iou(self, box1, box2):
        enclosing_box = np.concatenate(
            (np.minimum(box1[..., :2], box2[..., :2]),
             np.maximum(box1[..., 2:], box2[..., 2:]))
            , axis = -1
        )

        intersec_left = np.maximum(box1[..., 0], box2[..., 0])
        intersec_top = np.maximum(box1[..., 1], box2[..., 1])
        intersec_right = np.minimum(box1[..., 2], box2[..., 2])
        intersec_bottom = np.minimum(box1[..., 3], box2[..., 3])

        intersec_width = np.clip(intersec_right - intersec_left, a_min=0, a_max=None)
        intersec_height = np.clip(intersec_bottom - intersec_top, a_min=0, a_max=None)

        intersec_area = intersec_width * intersec_height
        enclosing_area = (enclosing_box[..., 2] - enclosing_box[..., 0]) * \
                         (enclosing_box[..., 3] - enclosing_box[..., 1])
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

        giou = 1. - (enclosing_area - (box1_area + box2_area - intersec_area)) / enclosing_area
        return giou


    def compute_reward(self, achieved_goal, desired_goal, info):
        assert desired_goal.shape == achieved_goal.shape

        if self.reward_type == "dense":
            # make sure that the reward range is [-1, 0]
            return self._get_generalized_iou(achieved_goal, desired_goal) - 1.
        else:
            min_achieved = achieved_goal[..., :2]
            max_achieved = achieved_goal[..., 2:]
            min_desired = desired_goal[..., :2]
            max_desired = desired_goal[..., 2:]
            comp_res = \
                np.concatenate((min_achieved > min_desired, max_achieved < max_desired), axis=-1).sum(-1)
            return - (comp_res.sum(-1) < 4).astype(np.float32)


    def step(self, action=None):
        self.frames = []

        start_pos = action[:2]
        ang = action[2]
        targ_pos = start_pos + 0.1 * np.asarray((np.cos(ang), np.sin(ang)))

        if action is not None:
            timeout = self.push_from_to(start_pos, targ_pos)

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout or timeout is None:
                obs = self._get_obs()
                achieved_goal = self._get_achieved_goal(obs)
                desired_goal = self.goal
                reward = self.compute_reward(achieved_goal, desired_goal, None)
                obs = {
                    "observation": obs.copy(),
                    "achieved_goal": achieved_goal.copy(),
                    "desired_goal": desired_goal.copy()
                }
                return obs, reward, True, self.info

        # Step simulator asynchronously until objects settle.
        if self._step_wait_until_settled:
            if self._verbal_frames:
                sim_step_acc = 0
                while not self.is_static:
                    p.stepSimulation()
                    sim_step_acc += 1
                    if sim_step_acc % self._verbal_frame_interval == 0:
                        self.frames.append(self.render("rgb_array"))
            else:
                while not self.is_static:
                    p.stepSimulation()

        obs = self._get_obs()
        achieved_goal = self._get_achieved_goal(obs)
        desired_goal = self.goal

        # Get task rewards.
        reward = self.compute_reward(achieved_goal, desired_goal, None)
        done = False

        info = {
            "is_success": reward == 0.
        }

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal.copy()
        }

        return obs, reward, done, info

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=5):
        """Move UR5 to target joint configuration."""
        t_acc = 0.
        sim_step_acc = 0
        while t_acc < timeout:
            t0 = time.time()
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            p.stepSimulation()
            t_acc += time.time() - t0
            sim_step_acc += 1
            if self._verbal_frames and sim_step_acc % self._verbal_frame_interval == 0:
                self.frames.append(self.render("rgb_array"))
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints