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

from a2c_ppo_acktr.envs import VecPyTorch, VecPyTorchFrameStack, VecPyTorchFrameStackDictObs, VecHistoryFrameStack, ShmemVecEnv
from .aai_config_generator import SingleConfigGenerator
from .aai_env_fixed import UnityEnvHeadless
from .preprocessors import GridOracle, GridOracleWithAngles, MetaObs, ObjectClassifier
import threading
import torch

def rotate(vec, angle):
    """rotates a vector on a given angle"""
    betta = np.radians(angle)
    x, y, z = vec
    x1 = np.cos(betta) * x - np.sin(betta) * z
    z1 = np.sin(betta) * x + np.cos(betta) * z
    return np.array([x1, y, z1])


class AnimalAIWrapper(gym.Env):

    ENV_RELOAD_PERIOD = 2400 #total update period( with num_processes==16) will be in range [2M, 6M] steps

    def __init__(self, env_path, rank, config_generator,
                 action_repeat=1, docker_training=False,
                 headless=False, image_only=True, channel_first=True,
                 reduced_actions=False, scale_reward=False):

        super(AnimalAIWrapper, self).__init__()
        #if config_generator is None we use random config!
        self.config_generator = config_generator if config_generator else SingleConfigGenerator()
        # all configs are copies of the config in the parent process, we need to make them different:
        self.config_generator.shuffle()
        self.image_only=image_only
        self.channel_first = channel_first

        self.num_episode = 0
        self.env = None
        # after self.env is closed socket is still in use for 60 seconds!
        # so instead of waiting for the old socket we open a new environment on another port:
        self._worker_id_pair = (rank, rank+200)
        self._env_args = dict(
            file_name=env_path, worker_id=rank,
            seed=rank, n_arenas=1,
            arenas_configurations=None,  # self.config,
            docker_training=docker_training,
            headless=headless
        )
        self._reload_env() #this creates new environment with self._env_args!
        #self.env = UnityEnvHeadless(**self._env_args)
        #self._set_config(self.config_generator.next_config())

        lookup_func = lambda a: {'Learner':np.array([a], dtype=float)}
        if reduced_actions:
            lookup = itertools.product([0,1], [0,1,2])
        else:
            lookup = itertools.product([0,1,2], repeat=2)

        lookup = dict(enumerate(map(lookup_func, lookup)))
        self.action_map = lambda a: lookup[a]
        
        self.observation_space = self._make_obs_space()
        self.action_space = Discrete(len(lookup))
        self.action_repeat = action_repeat

        self.scale_reward = scale_reward

        self.pos = np.zeros((3,), dtype=np.float32)
        self.angle = np.zeros((1,), dtype=np.float32)

        #print("Time limit: ", self.time_limit)

    def _reload_env(self):
        if self.env:
            self.env.close()
            del self.env

        #seed is changed to not repeat the same ENV_RELAOD_PERIOD episodes
        self._env_args['seed'] = (self._env_args['seed'] + self.num_episode) % 1009
        #worker id is changed to fix problem with linux sockets that wait for 60 seconds after closing
        self._env_args['worker_id'] = self._worker_id_pair[0]
        self._worker_id_pair = self._worker_id_pair[::-1]
        print("Reloading Env#{}: episode: {}, active_threads={}".format(
            self._env_args['worker_id'], self.num_episode, threading.active_count()
        ))
        # change UnityEnvHeadless to UnityEnvironment and remove headless arg
        # if you want to return to the animalai version of environemt
        env = UnityEnvHeadless(**self._env_args)
        self.env = env

    def _make_obs_space(self):
        img_shape = (3,84,84) if self.channel_first else (84,84,3)
        image_obs = space.Box(0.0, 255.0, img_shape, dtype=np.float32)
        if self.image_only:
            return image_obs
        #this low and high are scaled here! look at _make_obs method!
        speed_obs = space.Box(-2.4, 2.4, shape=[3,], dtype=np.float32)
        angle = space.Box(-1., 1., shape=[1,], dtype=np.float32)
        pos = space.Box(-10., 10., shape=[3,], dtype=np.float32)
        time = space.Box(0., 4., shape=[1,], dtype=np.float32)

        return space.Dict({
            "image":image_obs,
            "speed":speed_obs,
            'angle':angle,
            'pos':pos,
            'time': time,
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
            "angle":self.angle/180 - 1., #(0., 360.)/180 -1. --> (-1.,1.)
            "pos":self.pos/70., #(-700.,700.)/70. --> (-10.,10)
            "time": (self.time_limit - self.t)/250
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
        #print('ENV{}.reset(): starting new episode!'.format(self.env.port-5005))
        self.num_episode += 1
        if self.num_episode % self.ENV_RELOAD_PERIOD == 0:
            self._reload_env()

        self.t = 0
        self.angle = np.zeros((1,), dtype=np.float32)
        self.pos = np.zeros((3,), dtype=np.float32)

        self.ep_reward = 0.
        self.ep_success = False
        if self.scale_reward:
            self.scaled_ep_reward = 0.

        if forced_config:
            self._set_config(forced_config)
        else:
            self._set_config(self.config_generator.next_config())

        obs, r, done = self.process_state(self.env.reset(arenas_configurations=self.config))
        while done:
            obs, r, done = self.process_state(self.env.reset(arenas_configurations=self.config))

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

        if self.scale_reward:
            r = 0.3 * min(np.tanh(r), 0) + 5.0 * max(np.tanh(r), 0)
            self.scaled_ep_reward += r
            info["episode_scaled_reward"] = float(self.scaled_ep_reward)


        #if done:
            #print('ENV{}.step(): episode is done!'.format(self.env.port-5005))
            #obs = self.reset()

        return obs, r, done, info

    def close(self):
        self.env.close()


def make_env_aai(env_path, config_generator, rank,
                 headless=False,
                 grid_oracle_kwargs=None,
                 **kwargs):

    def _thunk():

        env = AnimalAIWrapper(env_path, rank, config_generator, headless=headless, **kwargs)

        if not kwargs.get('image_only', True):
            assert grid_oracle_kwargs is not None, \
                "If image_only==False, then we need grid oracle kwargs!"
            oracle_args = grid_oracle_kwargs.copy()
            oracle_type = oracle_args.pop('oracle_type')
            if oracle_type == "angles":
                # oracle_reward=1./100., penalty_mode=False,
                # trace_decay=1., exploration_only=False, num_angles=6
                env = GridOracleWithAngles(
                    **oracle_args
                ).wrap_env(env)
            elif oracle_type == "3d":
                # oracle_reward=2.5/100., penalty_mode=True,
                # trace_decay=0.999, exploration_only=False
                env = GridOracle(
                    **oracle_args,
                ).wrap_env(env)
            else:
                raise NotImplementedError("Only '3d' or 'angles' types for GridOracle")

            env = MetaObs().wrap_env(env)

        return env

    return _thunk

def make_vec_envs_aai(
        env_path, config_generator, seed, num_processes,
        device, num_frame_stack=None, headless=False,
        grid_oracle_kwargs=None, classifier_kwargs=None, **env_kwargs):

    envs = [make_env_aai(env_path, config_generator, i+seed,
                         headless, grid_oracle_kwargs,  **env_kwargs)
            for i in range(num_processes)]

    if len(envs) > 2:
        make_test = make_env_aai(env_path, config_generator,
                                 seed+num_processes+1, headless,
                                 grid_oracle_kwargs,**env_kwargs)
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

    if classifier_kwargs:
        envs = ObjectClassifier(**classifier_kwargs).wrap_env(envs)

    if num_frame_stack is not None:
        #VecHistoryFrameStack - stacks all observations in dict along a new dimention
        #envs = VecHistoryFrameStack(envs, num_frame_stack, device)

        if isinstance(envs.observation_space, gym.spaces.Box):
            envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        else:
            envs = VecPyTorchFrameStackDictObs(envs, num_frame_stack, device)

    #obs = envs.reset()
    return envs