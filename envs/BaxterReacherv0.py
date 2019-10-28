import copy
import os
import time
import sys
import rospy
import baxter_interface
import numpy as np
from baxter_pykdl import baxter_kinematics
from std_msgs.msg import (
    UInt16,
)
from baxter_interface import CHECK_VERSION
import rospkg
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from baxter_core_msgs.msg import JointCommand
from configs import BAXTER
from utils.rms import RunningMeanStd

from gym.spaces import Box

REAL_OBS_SPACE_LOW = np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059])
REAL_OBS_SPACE_HIGH = np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059])
REAL_ACT_SPACE_LOW = -np.array([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
REAL_ACT_SPACE_HIGH = np.array([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
EXECUTE_ACTION_FACTOR = BAXTER['execute_action_factor']
REAL_ACT_SPACE_HIGH *= EXECUTE_ACTION_FACTOR
REAL_ACT_SPACE_LOW *= EXECUTE_ACTION_FACTOR

LEFT_JOINT_NAMES = ("left_s0","left_s1","left_e0","left_e1","left_w0","left_w1","left_w2" )
RIGHT_JOINT_NAMES = ("right_s0","right_s1","right_e0","right_e1","right_w0","right_w1","right_w2")


def load_gazebo_model(model_path, model_name_in_gazebo = "model_0",
                      pose=Pose(position=Point(x=0., y=0., z=0.)),
                      reference_frame="world"):
    with open(model_path, "r") as block_file:
        block_xml = block_file.read().replace('\n', '')
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf(model_name_in_gazebo, block_xml, "/",
                               pose, reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models(model_name):
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model(model_name)
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

class BaxterReacherv0(object):
    """
    All communication between the algorithms and ROS is done through
    this class.
    """
    def __init__(self):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        rospy.init_node('baxter_reacher')
        self.dt = BAXTER['dt']
        self.RATE = 1. / self.dt
        self.using_torque =  BAXTER['using_torque']
        self.using_left_arm = BAXTER['using_left_arm']

        # init baxter interface
        self._llimb = baxter_interface.Limb("left")
        self._rlimb = baxter_interface.Limb("right")
        self.kin_left = baxter_kinematics('left')
        self.kin_right = baxter_kinematics("right")

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)
        self._pub_rate.publish(self.RATE)
        self.rate = rospy.Rate(self.RATE)

        self._max_time_step = 50
        self._step_counter = 0

        self.action_norm_factor = REAL_ACT_SPACE_HIGH

        # to control the orientation, we set 3 points on the gripper frame
        self.ee_points = np.array([[0., -0.04, 0.07], [0., 0.04, 0.07], [0., 0., 0.]])

        self.observation_space = Box(-np.inf * np.ones(10), np.inf * np.ones(10))
        self.action_space = Box(-np.ones(7,dtype=np.float32), np.ones(7,dtype=np.float32))

        self.indicator_flag = False

        self.num_envs = 1
        self.clipob = 10
        self.cliprew = 10

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = np.zeros(1)
        self.eprew = np.zeros(1)
        self.gamma = BAXTER["gamma"]
        self.epsilon = BAXTER["epsilon"]
        self._done = np.array(False).repeat(self.eprew.shape)
        self.prev_dones = np.array(False).repeat(self.eprew.shape)
        self._reward = np.zeros(1, dtype=np.float32)
        self._obs = np.zeros((1,)+self.observation_space.shape, dtype=np.float32)

    def norm_action(self, joint_vel):
        # to [-1., 1.]
        return joint_vel / self.action_norm_factor

    def unnorm_action(self, norm_joint_vel):
        return norm_joint_vel * self.action_norm_factor

    def reset(self):
        # if self.indicator_flag:
        #    delete_gazebo_models("indicator")
        joint_angles = np.random.rand(7) * (REAL_OBS_SPACE_HIGH - REAL_OBS_SPACE_LOW) + REAL_OBS_SPACE_LOW
        if self.using_left_arm:
            joint_angles = dict(zip(
                LEFT_JOINT_NAMES,
                joint_angles.tolist()
            ))
            self._llimb.move_to_joint_positions(joint_angles)
        else:
            joint_angles = dict(zip(
                RIGHT_JOINT_NAMES,
                joint_angles.tolist()
            ))
            self._rlimb.move_to_joint_positions(joint_angles)

        goal_r = np.random.uniform(0., 1.5)
        self.goal = 2. * np.random.rand(3) - 1.
        self.goal = self.goal / np.linalg.norm(self.goal) * goal_r
        self._step_counter = 0

        self._states = self.get_state()
        self._obs[0] = np.hstack([self._states['JOINT_ANGLES'],
                               # np.cos(self._states['JOINT_ANGLES']),
                               # np.sin(self._states['JOINT_ANGLES']),
                               # self._states['JOINT_VELOCITIES'],
                               # self.goal,
                               # self._states['END_EFFECTOR_POSITIONS'],
                               self._states['END_EFFECTOR_POSITIONS'] - self.goal])
        # model_path = os.path.join(rospkg.RosPack().get_path('baxter_rl'), "models", "block", "model.sdf")
        # load_gazebo_model(model_path, "indicator")
        self.indicator_flag = True
        self._done[0] = False
        return self._obfilt(self._obs)

    def step(self, action):
        """
        :param action: 7-d numpy array that defines joint velocities or torques.
        :return: observation (including 7 joint angles, velocities, targets, targets - endpoints)
        """
        if self.prev_dones[0]:
            self.reset()
            self.eprew[0] = 0.
        self._pub_rate.publish(self.RATE)
        action = action.squeeze()
        action = self.unnorm_action(action)
        if self.using_left_arm:
            self._action = dict(zip(LEFT_JOINT_NAMES, action))
            self._llimb.set_joint_velocities(self._action)
        else:
            self._action = dict(zip(RIGHT_JOINT_NAMES, action))
            self._rlimb.set_joint_velocities(self._action)
        self._states = self.get_state()
        self._obs[0] = np.hstack([self._states['JOINT_ANGLES'],
                               #np.cos(self._states['JOINT_ANGLES']),
                               #np.sin(self._states['JOINT_ANGLES']),
                               #self._states['JOINT_VELOCITIES'],
                               #self.goal,
                               #self._states['END_EFFECTOR_POSITIONS'],
                               self._states['END_EFFECTOR_POSITIONS'] - self.goal])
        self._reward_s = - np.linalg.norm(self._states['END_EFFECTOR_POSITIONS'] - self.goal)
        # self._reward_a = - np.square(action).sum() / 10.
        self._reward_s_log = - np.log(np.linalg.norm(self._states['END_EFFECTOR_POSITIONS'] - self.goal) + 1e-6)
        # self._reward = self._reward_a + self._reward_s + self._reward_s_log
        self._reward[0] = self._reward_s + self._reward_s_log
        self.eprew += self._reward

        self.ret = self.ret * self.gamma + self._reward
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            # r_out = r / sigma_ret (within [-cliprew, cliprew]) WHY????
            self._reward = np.clip(self._reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        self._step_counter += 1
        if self._step_counter >= self._max_time_step:
            self._done[0] = True
            self.prev_dones = self._done

        ########### IMPORTANT FOR DEFINATION OF dt !!!! ###########
        self.rate.sleep()
        return self._obfilt(self._obs), self._reward, self._done, {}

    def _obfilt(self, obs):
        # Normalize obs. (Minus mean and divided by sqrt(var).)
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def get_state(self):
        """Retrieves the state of the point mass"""
        if self.using_left_arm:
            _limb = self._llimb
            _kin = self.kin_left
            _arm = 'left'
        else:
            _limb = self._rlimb
            _kin = self.kin_right
            _arm = 'right'
        state = {}

        state['JOINT_ANGLES'] = np.array([
            _limb.joint_angles()[_arm+'_s0'],
            _limb.joint_angles()[_arm+'_s1'],
            _limb.joint_angles()[_arm+'_e0'],
            _limb.joint_angles()[_arm+'_e1'],
            _limb.joint_angles()[_arm+'_w0'],
            _limb.joint_angles()[_arm+'_w1'],
            _limb.joint_angles()[_arm+'_w2']
        ])

        state['JOINT_VELOCITIES'] = np.array([
            _limb.joint_velocities()[_arm+'_s0'],
            _limb.joint_velocities()[_arm+'_s1'],
            _limb.joint_velocities()[_arm+'_e0'],
            _limb.joint_velocities()[_arm+'_e1'],
            _limb.joint_velocities()[_arm+'_w0'],
            _limb.joint_velocities()[_arm+'_w1'],
            _limb.joint_velocities()[_arm+'_w2']
        ])

        # Points: (x,y,z) of the end effector
        state['END_EFFECTOR_POSITIONS'] = np.array(_limb.endpoint_pose()['position'])
        return state

    def render(self):
        pass

    def unwrapped(self):
        self._max_time_step = np.inf