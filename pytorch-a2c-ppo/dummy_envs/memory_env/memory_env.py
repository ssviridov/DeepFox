import numpy as np
import gym
import itertools

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from a2c_ppo_acktr.envs import ShmemVecEnv, VecPyTorch, VecHistoryFrameStack, VecPyTorchFrameStack

def make_dummy_memory_env( seed, episode_length, **kwargs):
    def _thunk():
        return DummyMemory(seed, length=episode_length, **kwargs)

    return _thunk


def make_vec_dummy_memory(
        num_processes, device, seed, episode_length, frame_stack=None, **env_kwargs
):

    envs = [make_dummy_memory_env(i+seed, episode_length, **env_kwargs)
            for i in range(num_processes)]

    if len(envs) > 2:
        make_test = make_dummy_memory_env(seed+num_processes+1, episode_length, **env_kwargs)
        test_env = make_test()

        spaces = (test_env.observation_space, test_env.action_space)
        print(test_env.observation_space, test_env.action_space)
        test_env.close()

        envs = ShmemVecEnv(envs, spaces)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if frame_stack is not None:
        envs = VecHistoryFrameStack(envs, frame_stack, device)
        #if isinstance(envs.observation_space, gym.spaces.Box):
        #    envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        #else:
        #    envs = VecPyTorchFrameStackDictObs(envs, num_frame_stack, device)

    return envs


class DummyMemory(gym.Env):
    INERNAL_STATE = ['left', 'right']
    OBS_DIM = 3

    def __init__(self, seed, prob=0.5, length=10):
        super(DummyMemory, self).__init__()

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            "obs":gym.spaces.Box(-1.,1., shape=(self.OBS_DIM,),dtype=np.float32),
        })

        self.length = length
        self.prob = prob
        self.obs_dim = self.OBS_DIM
        self.rnd = np.random.RandomState(seed)

    def step(self, action):

        self.t += 1
        if self.t >= self.length:
            if self._internal_state == "left":
                r = 1. if action == 0 else -1.
            elif self._internal_state == "right":
                r = 1. if action == 1 else -1.
            else:
                raise ValueError("internal_state should be in {}".format(self.INERNAL_STATE))

            obs = np.zeros(self.obs_dim, dtype=np.float32)
            return obs, r, True, {
                'episode_success':r == 1.,
                'episode_reward':r,
                'episode_len': self.t,
            }

        else:
            return self.make_obs(self.t), 0., False, {}

    def reset(self):
        self._internal_state = self.rnd.choice(self.INERNAL_STATE)
        self.t = 0
        return self.make_obs(self.t)

    def make_obs(self, t):
        time_mark = t/self.length
        obs = np.full(self.obs_dim, time_mark, dtype=np.float32)
        obs[0] = obs[-1] = 0.
        if t == 0:
            if self._internal_state == 'left':
                obs[0] = -1
            elif self._internal_state == 'right':
                obs[0] = 1.
            else:
                raise ValueError("Internal states should be either 'left' or 'right'!")
        return {"obs": obs}

    def close(self):
        del self.rnd


def read_action():
    action = input("Input Your Action: ")
    if action in ['0', '1']:
        print('act:', action)
        return int(action)
    else:
        action = np.random.choice([0,1])
        print('random act:', action)
        return action


def play_dummy_memory(num_episodes, episode_length):
    env = DummyMemory(17, length=episode_length)
    for i in range(num_episodes):
        obs = env.reset()
        print("\n====== EPISODE #{} ======".format(i))
        for t in itertools.count():
            print("===== STEP#{} =====".format(t))
            print("obs: ",obs['obs'])
            act = read_action()
            obs, r, done, info = env.step(act)
            print("r={:.1f}, done={}, info={}".format(r,done, info))
            if done: break


def check_dummy_memory(seed, num_episodes, episode_length):
    env = DummyMemory(seed, length=episode_length)
    count_left = 0
    count_success = 0
    for i in range(num_episodes):
        obs = env.reset()
        count_left += int(obs['obs'][0] == -1.)
        #print("\n====== EPISODE #{} ======".format(i))
        for t in itertools.count():
            #print("===== STEP#{} =====".format(t))
            #print("obs: ",obs['obs'])
            act = np.random.choice([0,1]) # read_action()
            obs, r, done, info = env.step(act)
            #print("r={:.1f}, done={}, info={}".format(r,done, info))
            if done:
                count_success += int(info['success'] == True)
                break

    print("Played {} episodes:")
    print("Generated {}/({:.1f}%) Left Configs".format(count_left, 100*count_left/num_episodes))
    print("Random policy success rate: {:.1f}%".format(100*count_success/num_episodes))


if __name__ == "__main__":
    #play_dummy_memory(10, 5)
    check_dummy_memory(33, 20000, 10)
