import animalai
from animalai.envs import UnityEnvironment, ArenaConfig
import numpy as np
import itertools

import gym
import gym.spaces as space
from gym.spaces.discrete import Discrete
import numpy as np
import os
import itertools

from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from a2c_ppo_acktr.envs import TransposeImage, VecPyTorch, VecPyTorchFrameStack, ShmemVecEnv
from .aai_config_generator import SingleConfigGenerator

from .aai_env_fixed import UnityEnvHeadless
import torch

def rotate(vec, angle):
    """rotates a vector on a given angle"""
    betta = np.radians(angle)
    x, y, z = vec
    x1 = np.cos(betta) * x - np.sin(betta) * z
    z1 = np.sin(betta) * x + np.cos(betta) * z
    return np.array([x1, y, z1])


class AnimalAIWrapper(gym.Env):
    def __init__(self, env_path, rank, config_generator,
                 action_repeat=1, docker_training=False,
                 headless=False, image_only=True, channel_first=True):
        super(AnimalAIWrapper, self).__init__()
        #if config_generator is None we use random config!
        self.config_generator = config_generator if config_generator else SingleConfigGenerator()
        # all configs are copies of the config in the parent process, we need to make them different:
        self.config_generator.shuffle()
        self.image_only=image_only
        self.channel_first = channel_first

        self._set_config(self.config_generator.next_config())
        #change UnityEnvHeadless to UnityEnvironment and remove headless arg
        # if you want to return to the animalai version of environemt
        #self.env = UnityEnvironment(
        self.env = UnityEnvHeadless( #
            file_name=env_path, worker_id=rank,
            seed=rank, n_arenas=1,
            arenas_configurations=self.config,
            docker_training=docker_training,
            headless=headless
        )

        lookup_func = lambda a: {'Learner':np.array([a], dtype=float)}
        #if reduced_actions:
        #    lookup = itertools.product([0,1], [0,1,2])
        #else:
        lookup = itertools.product([0,1,2], repeat=2)
        lookup = dict(enumerate(map(lookup_func, lookup)))
        self.action_map = lambda a: lookup[a]
        
        self.observation_space = self._make_obs_space()
        self.action_space = Discrete(len(lookup))
        self.action_repeat = action_repeat

        self.pos = np.zeros((3,), dtype=np.float32)
        self.angle = np.zeros((1,), dtype=np.float32)

        print("Time limit: ", self.time_limit)

    def _make_obs_space(self):
        img_shape = (3,84,84) if self.channel_first else (84,84,3)
        image_obs = space.Box(0.0, 255.0, img_shape, dtype=np.float32)
        if self.image_only:
            return image_obs
        #this low and high are scaled here! look at _make_obs method!
        speed_obs = space.Box(-2.4, 2.4, shape=[3,], dtype=np.float32)
        angle = space.Box(-1., 1., shape=[1,], dtype=np.float32)
        pos = space.Box(-10., 10., shape=[3,], dtype=np.float32)

        return space.Dict({
            "image":image_obs,
            "speed":speed_obs,
            'angle':angle,
            'pos':pos
        })

    def _set_config(self, new_config):
        self.config_name = new_config['config_name']
        self.config = new_config['config']

        self.time_limit = self.config.arenas[0].t

    def process_state(self, state):
        obs = self._make_obs(state)
        r = state['Learner'].rewards[0]
        done = state['Learner'].local_done[0]
        return obs, r, done

    def _make_obs(self, state):
        img = state['Learner'].visual_observations[0][0]*255.
        if self.channel_first: #C,H,W format for pytorch conv layers
            img = img.transpose(2, 0, 1)
        if self.image_only: return img

        speed = state['Learner'].vector_observations[0]
        absolute_speed = rotate(speed, self.angle[0])
        self.pos[:] += absolute_speed
        return {
            "image":img,
            "speed":absolute_speed /10., #(-21.,21.)/10. --> (-2.1,2.1)
            "angle":self.angle/180 -1., #(0., 360.)/180 -1. --> (-1.,1.)
            "pos":self.pos/70. #(-700.,700.)/70. --> (-10.,10)
        }

    def _make_info(self, obs, r, done):
        if done:
            return {
                "episode_reward": float(self.ep_reward),
                'episode_success': r > 0.4,
                'episode_len': int(self.t),
            } #, "config":self.config}
        else:
            return dict()

    def reset(self, forced_config=None):
        if forced_config:
            self._set_config(forced_config)
        else:
            self._set_config(self.config_generator.next_config())

        obs, r, done = self.process_state(self.env.reset(arenas_configurations=self.config))
        while done:
            obs, r, done = self.process_state(self.env.reset(arenas_configurations=self.config))

        self.t = 0
        self.angle = np.zeros((1,), dtype=np.float32)
        self.pos = np.zeros((3,), dtype=np.float32)

        self.ep_reward = 0.
        self.ep_success = False

        return obs

    def _update_angle(self, action):
        if (action - 1) % 3 == 0: #TURN_RIGHT:
            self.angle[0] -= 6
        if (action - 2) % 3 == 0: #TURN_LEFT
            self.angle[0] += 6
        self.angle[0] = self.angle[0] % 360

    def step(self, action):
        r = 0
        for i in range(self.action_repeat):
            self._update_angle(action)
            obs, r_, done = self.process_state(
                self.env.step(vector_action=self.action_map(action))
            )
            r += r_
            self.t += 1
            done = done or self.t >= self.time_limit
            if done: 
                break

        self.ep_reward += r
        info = self._make_info(obs, r, done)

        if done:
            obs = self.reset()

        return obs, r, done, info

    def close(self):
        self.env.close()


class GridBasedExploration(gym.Wrapper):

    def __init__(
        self, env,
        visiting_r=1./100.,
        grid_size=(31,5,31), # we start at the center of X, and Z, dimensions and at the bottom of Y
        cell_size=(1.,1./2, 1.),
        observe_map=False,
        trace_decay=1.,  # this means no decay!
        revisit_threshold = 0.01
    ):
        assert isinstance(env.observation_space, space.Dict), "This wrapper use obs['pos']!"
        super(GridBasedExploration, self).__init__(env)
        self.grid_size=np.array(grid_size)
        self.cell_size=np.array(cell_size)
        self.visiting_r = visiting_r
        self.observe_map = observe_map
        self._visited = np.zeros(grid_size, dtype=np.float32)
        self.revisit_threshold = revisit_threshold
        self.total_expl_r = 0.

    def set_start_coords(self):
        self.start_x = grid_size[0] // 2
        self.start_y = 0 # Y is a vertical dimension but we start slightly bellow the ground somehow
        self.start_z = grid_size[2] // 2

    def reset(self, **kwargs):
        self._visited[:] = 0
        self.set_start_coords()
        self._visited[x,y,z] = 1.
        obs = self.env.reset(**kwargs)
        return self._observation(obs)

    def _observation(self):
        if self.observe_map:
            obs['visited'] = self._visited.copy()
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        if self.trace_decay<1.:
            self.map *= self.trace_decay
        cell_pos = obs['pos']/self.cell_size
        x, y, z = np.round(cell_pos).astype(np.int32)
        r = self.visit(x,y,z)
        obs = self._observation(obs)
        self._fill_info(obs, r, done, info)

        return obs, r, done, info

    def _fill_info(self, obs, r, done, info):
        self.total_expl_r += r
        if self.visiting_r > 0.:
            info['expl_r'] = r
            if done:
                info['episode_expl_r'] = self.total_expl_r

    def visit(self, x,y,z):
        _x += self.start_x
        _y += self.start_y
        _z += self.start_z
        r = self.visiting_r if self._visited[_x,_y,_z] < self.revisit_threshold else 0.
        self._map[_x,_y,_z] = 1.
        return r


def make_env_aai(env_path, config_generator, rank, log_dir, allow_early_resets, headless=False, **kwargs):
    if env_path is None:
        env_path = 'aai_resources/env/AnimalAI.x86_64'
    def _thunk():
        env = AnimalAIWrapper(env_path, rank, config_generator, headless=headless, **kwargs)
        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
        # env = TransposeImage(env, op=[2, 0, 1])
        return env

    return _thunk


def make_vec_envs_aai(env_path, config_generator, seed, num_processes, log_dir, device, allow_early_resets,
                      num_frame_stack=None, headless=False, **env_kwargs):

    envs = [make_env_aai(env_path, config_generator,
                         i+seed, log_dir, allow_early_resets, headless=headless, **env_kwargs)
            for i in range(num_processes)]

    if len(envs) > 2:
        make_test = make_env_aai(env_path, config_generator, seed+num_processes+1,
                                 log_dir, allow_early_resets, headless=headless, **env_kwargs)
        test_env = make_test()
        spaces = (test_env.observation_space, test_env.action_space)
        print(test_env.observation_space, test_env.action_space)
        test_env.close()

        envs = ShmemVecEnv(envs, spaces)
        # envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    #obs = envs.reset()
    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStackDictObs(envs, num_frame_stack, device)
    elif isinstance(envs.observation_space, gym.spaces.Box):
        envs = VecPyTorchFrameStack(envs, 2, device)

    #obs = envs.reset()
    return envs