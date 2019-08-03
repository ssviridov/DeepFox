import animalai
from animalai.envs import UnityEnvironment, ArenaConfig
import numpy as np
import itertools

import gym
from gym.spaces.box import Box
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

class AnimalAIWrapper(gym.Env):
    def __init__(self, env_path, rank, config_generator, time_reward=0.0,
                 action_repeat=1, reduced_actions=False, docker_training=False,
                 use_info=False, headless=False):
        super(AnimalAIWrapper, self).__init__()
        #if config_generator is None we use random config!
        self.config_generator = config_generator if config_generator else SingleConfigGenerator()
        self.use_info = use_info

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
        if reduced_actions:
            lookup = itertools.product([0,1], [0,1,2])
        else:
            lookup = itertools.product([0,1,2], repeat=2)
        lookup = dict(enumerate(map(lookup_func, lookup)))
        self.action_map = lambda a: lookup[a]
        
        self.observation_space = Box(0.0, 255.0, [84,84,3], dtype=np.float32)
        self.action_space = Discrete(len(lookup))
        self.time_reward = time_reward
        self.action_repeat = action_repeat
        
        print("Time limit: ", self.time_limit)

    def _set_config(self, new_config):
        self.config = new_config
        self.time_limit = self.config.arenas[0].t

    def process_state(self, state):
        img = 255*state['Learner'].visual_observations[0][0]
        vec = state['Learner'].vector_observations[0]
        r = state['Learner'].rewards[0]
        done = state['Learner'].local_done[0]
        return img, vec, r, done

    def _make_info(self, img, vec, r, done):
        return {"vec":vec, "r":r, "config":self.config}

    def reset(self):
        self._set_config(self.config_generator.next_config())

        img, vec, r, done = self.process_state(self.env.reset(arenas_configurations=self.config))
        while done:
            img, vec, r, done = self.process_state(self.env.reset(arenas_configurations=self.config))
        self.t = 0
        if self.use_info:
            info = self._make_info(img, vec, r, done)
            return img, info
        else:
            return img
    
    def step(self,action):
        r = 0
        for i in range(self.action_repeat):
            obs, vec, r_, done = self.process_state(self.env.step(vector_action=self.action_map(action)))
            r += r_ - self.time_reward
            self.t += 1
            done = done or self.t >= self.time_limit
            if done: 
                break

        info = self._make_info(obs, vec, r, done) if self.use_info else {}

        if done:
            if self.use_info:
                obs, info = self.reset()
            else:
                obs = self.reset()

        return obs, r, done, info


def make_env_aai(env_path, config_generator, rank, log_dir, allow_early_resets, headless=False, **kwargs):
    if env_path is None:
        env_path = 'aai_resources/env/AnimalAI.x86_64'
    def _thunk():
        env = AnimalAIWrapper(env_path, rank, config_generator, headless=headless, **kwargs)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
        env = TransposeImage(env, op=[2, 0, 1])
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

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 2, device)

    return envs