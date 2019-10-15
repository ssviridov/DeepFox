import gym
import gym_minigrid

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from a2c_ppo_acktr.envs import ShmemVecEnv, VecPyTorch, VecHistoryFrameStack, VecPyTorchFrameStack, VecPyTorchFrameStackDictObs


class MinigridWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._update_obs_space()
        self._episode_reward = 0.
        self._t = 0

    def _update_obs_space(self):
        old_image = self.observation_space.spaces['image']
        H, W, C = old_image.shape

        new_image = gym.spaces.Box(
            low=old_image.low.min(),
            high=old_image.high.max(),
            shape=(C, H, W),
            dtype=old_image.dtype)

        self.observation_space.spaces['image'] = new_image

    def reset(self, **kwargs):
        self._episode_reward = 0.
        self._t = 0
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):

        obs, r, done, info = self.env.step(action)
        self._episode_reward += r
        self._t += 1

        if done:
            info['episode_success'] = r >= 0.2
            info['episode_len'] = self._t
            info['episode_reward'] = self._episode_reward

        return self.observation(obs), r, done, info

    def observation(self, observation):
        image = observation['image']
        return {"image":image.transpose(2, 0, 1)}


def make_minigrid(name, seed):

    def create():
        env = gym.make(name)
        env = MinigridWrapper(env)
        env.seed(seed)
        return env

    return create


def make_vec_minigrid(name, num_processes, seed, device, frame_stack=None, image_only_stack=True):
    envs = [make_minigrid(name, i + seed)
            for i in range(num_processes)]

    if len(envs) > 2:
        make_test = make_minigrid(name, seed + num_processes + 1)
        test_env = make_test()

        spaces = (test_env.observation_space, test_env.action_space)
        print(test_env.observation_space, test_env.action_space)
        test_env.close()

        envs = ShmemVecEnv(envs, spaces)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if frame_stack > 1:
        if image_only_stack:
            envs = VecPyTorchFrameStackDictObs(envs, frame_stack, device)
        else:
            envs = VecHistoryFrameStack(envs, frame_stack, device)
        # if isinstance(envs.observation_space, gym.spaces.Box):
        #    envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        # else:
        #

    return envs