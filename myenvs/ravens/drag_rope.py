import pybullet as p
import numpy as np
import os
import gym
from gym import spaces

from ravens.environments.environment import *
from ravens.tasks.task import Task
from ravens.utils import pybullet_utils, utils

class ManipulatingRope(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.pos_eps = 0.02

    def reset(self, env):
        super().reset(env)

        n_parts = 20
        radius = 0.005
        length = 2 * radius * n_parts * np.sqrt(2)

        # Add 3-sided square.
        square_size = (length, length, 0)
        square_pose = self.get_random_pose(env, square_size)
        square_template = 'square/square-template.urdf'
        replace = {'DIM': (length,), 'HALF': (length / 2 - 0.005,)}
        urdf = self.fill_template(square_template, replace)
        env.add_object(urdf, square_pose, 'fixed')
        os.remove(urdf)

        # Get corner points of square.
        corner0 = (length / 2, length / 2, 0.001)
        corner1 = (-length / 2, length / 2, 0.001)
        corner0 = utils.apply(square_pose, corner0)
        corner1 = utils.apply(square_pose, corner1)

        # Add cable (series of articulated small blocks).
        increment = (np.float32(corner1) - np.float32(corner0)) / n_parts
        position = (0.49, 0.11, 0.05)
        position = np.float32(position)
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)
        parent_id = -1
        targets = []
        objects = []
        for i in range(n_parts):
            position[2] += np.linalg.norm(increment)
            part_id = p.createMultiBody(0.1, part_shape, part_visual,
                                        basePosition=position)
            if parent_id > -1:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=parent_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, np.linalg.norm(increment)),
                    childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)
            if (i > 0) and (i < n_parts - 1):
                color = utils.COLORS['red'] + [1]
                p.changeVisualShape(part_id, -1, rgbaColor=color)
            env.obj_ids['rigid'].append(part_id)
            parent_id = part_id
            target_xyz = np.float32(corner0) + i * increment + increment / 2
            objects.append((part_id, (0, None)))
            targets.append((target_xyz, (0, 0, 0, 1)))

        matches = np.clip(np.eye(n_parts) + np.eye(n_parts)[::-1], 0, 1)
        self.goals.append((objects, matches, targets,
                           False, False, 'pose', None, 1))
        for i in range(480):
            p.stepSimulation()


class DragRopeEnv(Environment):

    def __init__(self, reward_type="sparse"):
        super(DragRopeEnv, self).__init__(assets_root="ravensenvs/ravens/environments/assets", disp=False,
                                             shared_memory=False, hz=480)
        task = ManipulatingRope()
        self.set_task(task)

        self.speed = 0.005
        self.dist_thresh = 0.05
        self.reward_type = reward_type
        self.key_point_num = 8
        self._rope_symmetry = False
        self._step_wait_until_settled = False
        self._verbose_frames = False
        self._verbose_frame_interval = 12
        self.frames = []
        self.height = None

        obs = self.reset()
        self.action_space = gym.spaces.Box(-0.03, 0.03, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation":spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
        })


    def init_pick(self, pick_pose):
        # Execute picking primitive.
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        prepick_pose = utils.multiply(pick_pose, prepick_to_pick)

        timeout = True
        while timeout:
            timeout = self.movep(prepick_pose, self.speed)

        # Move towards pick pose until contact is detected.
        delta = (np.float32([0, 0, -0.001]),
               utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose
        while not self.ee.detect_contact():  # and target_pose[2] > 0:
            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= self.movep(targ_pose)
            if timeout:
                return True

        self.height = p.getLinkState(self.ur5, self.ee_tip)[0][2] + 0.01

        # Activate end effector, move up, and check picking success.
        self.ee.activate()
        pick_success = self.ee.check_grasp()
        return pick_success

    def drag_to(self, place_pose):
        self.picked = self.ee.check_grasp()
        # Execute placing primitive if pick is successful.
        timeout = False
        if self.picked:
            timeout |= self.movep(place_pose, self.speed)
            if timeout:
                return True
        else:
            print("No grasp detected...")
            return None
        return timeout


    def _get_keypoint_ids(self):
        num = len(self.obj_ids["rigid"])
        indices = [0]
        n_mid_points = self.key_point_num - 2
        interval = (num - 2) // n_mid_points
        for i in range(1, 1 + n_mid_points):
            indices.append(i * interval)
        indices.append(num - 1)
        return [self.obj_ids["rigid"][i] for i in indices]


    def _get_obs(self):
        obj_states = []
        for obj_id in self.key_point_indices:
            obj_states.append(p.getBasePositionAndOrientation(obj_id)[0])
            obj_states.append(p.getBaseVelocity(obj_id)[0])
        obj_states = np.concatenate(obj_states)
        robot_state = [p.getJointState(self.ur5, i) for i in self.joints]
        robot_state = np.concatenate([(s[0], s[1]) for s in robot_state])
        ee_state = p.getLinkState(self.ur5, self.ee_tip)
        return np.concatenate([obj_states, robot_state, ee_state[0]])


    def _get_achieved_goal(self):
        rope_particle_ids = (self.task.goals[0][0][0][0], self.task.goals[0][0][-1][0])
        end_particle_pos = [p.getBasePositionAndOrientation(i)[0][:2] for i in rope_particle_ids]
        return np.concatenate(end_particle_pos)


    def reset(self):
        """Performs common reset functionality for all supported tasks."""

        while True:
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
            self.goal = self.task.goals[-1][2]
            self.goal = np.concatenate((self.goal[0][0][:2], self.goal[-1][0][:2]))

            # Re-enable rendering.
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            self.key_point_indices = self._get_keypoint_ids()

            # grasp one end of the rope
            end_points = self._get_achieved_goal()[:3]
            pick_pose = (np.asarray(end_points), np.asarray((0, 0, 0, 1)))
            picked = self.init_pick(pick_pose)
            if not picked:
                print("Object not successfully picked. Trying to reset again...")
                break

            observation = self._get_obs()
            achieved_goal = self._get_achieved_goal()
            obs = {
                "observation": observation.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy()
            }

            return obs


    def compute_reward(self, achieved_goal, desired_goal, info):
        _shape = achieved_goal.shape[:-1] + (-1, 2)
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
        self.frames = []
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_pos = np.zeros(3)
        delta_pos[:2] = action
        targ_pos = [p.getLinkState(self.ur5, self.ee_tip)[0], p.getLinkState(self.ur5, self.ee_tip)[1]]
        targ_pos[0] = np.asarray(targ_pos[0]) + delta_pos
        targ_pos[0][2] = self.height
        targ_pos[0] = tuple(targ_pos[0])

        if action is not None:
            timeout = self.drag_to(targ_pos)

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout or timeout is None:
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
        if self._step_wait_until_settled:
            if self._verbose_frames:
                sim_step_acc = 0
                while not self.is_static:
                    p.stepSimulation()
                    sim_step_acc += 1
                    if sim_step_acc % self._verbose_frame_interval == 0:
                        self.frames.append(self.render("rgb_array"))
            else:
                while not self.is_static:
                    p.stepSimulation()

        obs = self._get_obs()
        achieved_goal = self._get_achieved_goal()
        desired_goal = self.goal

        # Get task rewards.
        reward = self.compute_reward(achieved_goal, desired_goal, None)
        done = False

        info = {
            "is_success": reward >= - self.dist_thresh
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
            if self._verbose_frames and sim_step_acc % self._verbose_frame_interval == 0:
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

