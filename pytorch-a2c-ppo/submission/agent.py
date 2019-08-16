import yaml
from animalai_train.trainers.ppo.policy import PPOPolicy
from animalai.envs.brain import BrainParameters
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from collections import deque, defaultdict
import torch as th
import itertools as it
import numpy as np
DOCKER_CONFIG_PATH = '/aaio/data/sub_config.yaml'


class ActionAdapter(object):
    def __init__(
            self,
            unflatten=True,
            available_moves=(0,1,2),
            available_turns=(0,1,2)
    ):
        self.unflatten = unflatten
        lookup_func = lambda a:{'Learner':np.array([a], dtype=float)}
        lookup = it.product(available_moves, available_turns)
        self.lookup = dict(enumerate(map(lookup_func, lookup)))

    def __call__(self, action):
        if not self.unflatten:
            return action

        try:
            len(action)
        except TypeError as e:
            return self.lookup[action]
        else:
            return [self.lookup[a] for a in action]

    def reset(self):
        pass


class ObservationAdapter(object):

    def __init__(
            self,
            device,
            nstack=2,
            image_only=True,
            transpose=True,
            unsqueeze=True
    ):
        self.device = device
        self.image_only = image_only
        self.transpose = transpose
        self.unsqueeze = unsqueeze
        self.nstack = nstack
        self.stacked_obs = defaultdict(lambda :deque(maxlen=nstack))

    def reset(self):
        self.stacked_obs.clear()

    def __call__(self, obs):
        if isinstance(obs, tuple):
            if self.image_only:
                return self.to_torch(0, obs[0], True)

            return tuple( # we assume that image goes first in these tuples
                self.to_torch(i, v, is_image=(i==0))
                for i, v in enumerate(obs)
            )

        elif isinstance(obs, dict):
            if self.image_only:
                return self.to_torch('image', obs['image'], True)

            return {  # we assume that image goes first in these tuples
                k:self.to_torch(k, v, is_image=(k=="image"))
                for k, v in obs.items()
            }

    def stack_vars(self, key, var):
        self.stacked_obs[key].append(var)
        while len(self.stacked_obs[key]) < self.nstack:
            self.stacked_obs[key].append(var)

        return th.cat(tuple(self.stacked_obs[key]), dim=1)

    def to_torch(self, key, var, is_image=False):
        if is_image:
            if self.unsqueeze:
                var = var.transpose(2,0,1)  # (H,W,C)-> (C,H,W)
            else:
                var = var.transpose(0, 3,1,2) #(batch, H,W,C)-> (batch,C,H,W)
            var *= 255.
        var = th.tensor(var, dtype=th.float32, device=self.device)
        if self.unsqueeze:
            var.unsqueeze_(0)
            if len(var.shape) < 2:
                var.unsqueeze_(1)

        return self.stack_vars(key, var)


class Agent(object):

    def __init__(self, config_path=DOCKER_CONFIG_PATH):
        """
         Load your agent here and initialize anything needed
        """

        # Load the configuration and model using ABSOLUTE PATHS
        self.eval_config_path = config_path
        with open(self.eval_config_path) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        self.device = th.device(
            self.config['device'] if th.cuda.is_available() else "cpu"
        )

        data = th.load(
            self.config['model_path'],
            map_location=self.device
        )
        self.model = data['model'] if isinstance(data, dict) else data[0]

        self.model.to(self.device)
        self.greedy = self.config['greedy_policy']

        self.rnn_state = None
        self.masks = None
        self.episode_is_running = False
        self.adapt_act = None

        self.adapt_act = ActionAdapter(
            **self.config.get('action_adapter', {})
        )
        self.adapt_obs = ObservationAdapter(
            self.device,
            **self.config.get("observation_adapter",{})
        )

        self.report_config()

    def report_config(self):
        print("current device:", self.device)
        print("greedy:", self.greedy)
        print("model:", self.config['model_path'])
        print()

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.episode_is_running = False
        self.rnn_state = None
        self.masks = None
        self.adapt_obs.reset()
        self.adapt_act.reset()


    def _start_episode(self, batch_size=1):

        self.rnn_state = th.zeros(
            batch_size,
            self.model.recurrent_hidden_state_size,
            device=self.device
        )
        self.masks = th.zeros(batch_size,1, device=self.device)
        self.episode_is_running = True

    def step(self, obs, reward, done, info):
        """
        A single step the agent should take based on the current
        :param brain_info:  a single BrainInfo containing the observations and reward for a single step for one agent
        :return:            a list of actions to execute (of size 2)
        """
        with th.no_grad():

            if not self.episode_is_running:
                self.batch_size = self._get_batch_size(reward)
                self._start_episode(self.batch_size)

            obs = self.adapt_obs(obs)

            _, action, _, self.rnn_state = self.model.act(
                obs, self.rnn_state,
                self.masks, deterministic=self.greedy
            )

            if self.batch_size == 1:
                action = action.squeeze()

            action = action.tolist()
            action = self.adapt_act(action)

            if done:
                self.episode_is_running = False

        return action

    def _get_batch_size(self, var):
        if np.isscalar(var) or len(var) == 1:
            return 1
        else:
            return len(var)


def create_env(seed=None):

    arena_config = ArenaConfig(
        "aai_resources/default_configs/1-Food.yaml"
    )

    seed = seed if seed else rnd.randint(0, 1000)
    env = AnimalAIEnv(
        environment_filename="aai_resources/env/AnimalAI",
        worker_id=seed,
        n_arenas=1,
        arenas_configurations=arena_config,
        docker_training=False,
        retro=False
    )
    return env


if __name__ == "__main__":
    import random as rnd
    import itertools as it

    agent = Agent('submission/data/sub_config.yaml')
    env = create_env()

    obs = env.reset()


    print('Running 5 episodes')
    for k in range(5):
        cumulated_reward = 0
        print('Episode {} starting'.format(k))
        try:
            agent.reset()
            #obs: tuple(84,84,3),(3,), reward: int, done: bool, info: dict{"brain_info":..., ..}
            obs, reward, done, info = env.step([0, 0])

            for step in it.count(1):
                action = agent.step(obs, reward, done, info)
                obs, reward, done, info = env.step(action)
                cumulated_reward += reward
                if done:
                    break
        except Exception as e:
            print('Episode {} failed'.format(k))
            raise e

        print(
            'Episode {0} completed, reward {1:0.2f}, num_steps {2}'.format(
                k, cumulated_reward, step
            ))

    print('SUCCESS')