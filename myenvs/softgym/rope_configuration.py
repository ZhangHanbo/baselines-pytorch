import numpy as np
import pickle
import os
import os.path as osp
import cv2
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place, center_object
from gym import spaces
import time

class RopeConfigurationEnv(RopeNewEnv):
    def __init__(self, reward_type = "sparse", cached_states_path='rope_configuration_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_distance_diff = None
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        self.dist_thresh = 0.05
        self.reward_type = reward_type

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True
        self._rope_symmetry = False

        self._goal_save_dir = "save/rope_configuration/goals/"
        if not osp.exists("save/rope_configuration/goals/"):
            os.makedirs("save/rope_configuration/goals/")

        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        if config is None:
            config = self.get_default_config()
        default_config = config
        config = deepcopy(default_config)
        config['segment'] = 40
        self.set_scene(config)

        self.update_camera('default_camera', default_config['camera_params']['default_camera'])
        config['camera_params'] = deepcopy(self.camera_params)
        self.action_tool.reset([0., -1., 0.])

        # random_pick_and_place(pick_num=4, pick_scale=0.005)
        center_object()
        generated_configs = deepcopy(config)
        print('config: {}'.format(config['camera_params']))
        generated_states = deepcopy(self.get_state())

        return [generated_configs] * num_variations, [generated_states] * num_variations

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Reward is the distance between the endpoints of the rope"""
        _shape = achieved_goal.shape[:-1] + (len(self.key_point_indices), 3)
        achieved_goal = achieved_goal.reshape(_shape)
        desired_goal = desired_goal.reshape(_shape)

        if self._rope_symmetry:
            # the symmetry of the rope
            dist1 = np.linalg.norm(achieved_goal - desired_goal, axis=-1).mean(-1)
            dist2 = np.linalg.norm(np.flip(achieved_goal, axis=-2) - desired_goal, axis=-1).mean(-1)
            dist = np.minimum(dist1, dist2)
        else:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1).mean(-1)

        if self.reward_type == "sparse":
            return - (dist >= self.dist_thresh).astype(np.float32)
        else:
            return - dist

    def reset(self):
        self.goal = self._sample_goal()

        self.current_config = self.cached_configs[0]
        self.set_scene(self.cached_configs[0], self.cached_init_states[0])
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0
        obs = self._reset()
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))

        n_key_points = len(self.key_point_indices)
        achieved_goal = obs.reshape(-1, 3)[:n_key_points].flatten()

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def step(self, action):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        for i in range(self.action_repeat):
            self._step(action)
            if i % 2 == 0:  # No need to record each step
                frames.append(self.get_image())

        obs = self._get_obs()

        achived_goal = obs.reshape(-1, 3)[:len(self.key_point_indices)].flatten()
        desired_goal = self.goal
        reward = self.compute_reward(achived_goal, desired_goal, None)

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achived_goal.copy(),
            "desired_goal": desired_goal.copy()
        }
        info = {'is_success': reward >= - self.dist_thresh}

        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        info['flex_env_recorded_frames'] = frames
        return obs, reward, done, info

    def _sample_goal(self, save_goal_img=False):
        # reset scene
        config = self.cached_configs[0]
        init_state = self.cached_init_states[0]
        self.set_scene(config, init_state)
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        # randomize the goal.
        random_pick_and_place(pick_num=4, pick_scale=0.005)
        center_object()

        # read goal state
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.key_point_indices, :3]
        goal = keypoint_pos.flatten()

        # visualize the goal scene
        if save_goal_img:
            # rendering needs the action_tool being initialized.
            if hasattr(self, 'action_tool'):
                self.action_tool.reset([10, 10, 10])
            save_name = str(time.time())
            goal_img = self.render(mode='rgb_array')
            cv2.imwrite(osp.join(self._goal_save_dir, save_name + ".png"), goal_img[:, :, ::-1])

        return goal

    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # set reward range
        self.reward_max = 0
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])

        # set reward range
        self.reward_max = 0
        self.reward_min = -self.rope_length
        self.reward_range = self.reward_max - self.reward_min

        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return


