import yaml
from animalai_train.trainers.ppo.policy import PPOPolicy
from animalai.envs.brain import BrainParameters
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from collections import deque, defaultdict
import os.path as ospath
import json
import torch as th
import itertools as it
import numpy as np
from a2c_ppo_acktr.preprocessors import GridOracle, GridOracleWithAngles, MetaObs
#change this path to specify a model you want to submit:
DOCKER_CONFIG_PATH = '/aaio/data/pretrained/default2-reduced-actions/sub_config.yaml'


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
            frame_stack=1,
            image_only=True,
            transpose=True,
            unsqueeze=True,
            stackables=(),
    ):
        self.device = device
        self.image_only = image_only
        self.transpose = transpose
        self.unsqueeze = unsqueeze
        self.nstack = frame_stack
        self.stacked_obs = defaultdict(lambda :deque(maxlen=self.nstack))
        self.stackables = stackables

    def reset(self, time_limit):
        self.stacked_obs.clear()

    def __call__(self, prev_action, obs, r, done, info):

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
        if key not in self.stackables or self.nstack == 1:
            return var

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


class ExtraObsAdapter(ObservationAdapter):

    def __init__(self, *args, **kwargs):
        oracle_args = kwargs.pop("grid_oracle", {})
        oracle_type = oracle_args.pop("oracle_type")

        if oracle_type == "angles":
            grid_oracle = GridOracleWithAngles(
                **oracle_args
            )
        elif oracle_type == "3d":
            grid_oracle= GridOracle(
                **oracle_args
            )
        else:
            raise NotImplementedError()

        self.preprocessors = [grid_oracle, MetaObs()]

        super(ExtraObsAdapter, self).__init__(*args, **kwargs)

    def _rotate_XZ(self, vec, angle):
        """rotates a vector on a given angle"""
        betta = np.radians(angle)
        x, y, z = vec
        x1 = np.cos(betta) * x - np.sin(betta) * z
        z1 = np.sin(betta) * x + np.cos(betta) * z
        return np.array([x1, y, z1])

    def reset(self, time_limit):
        super(ExtraObsAdapter, self).reset(time_limit)
        self.angle = np.zeros((1,), dtype=np.float32)
        self.pos = np.zeros((3,), dtype=np.float32)
        self.time_limit = time_limit
        self.t = 0

        print("Check that your preprocessor is OK with empty/meaningless"
              " input to reset method")
        #yeah if we feed GridOracle with an empty obs nothing bad happens
        #but future preprocessors could brake with this...
        obs = {}
        for p in self.preprocessors:
            obs = p.reset(obs)

    def _update_angle(self, action):
        if (action - 1) % 3 == 0: #TURN_RIGHT:
            self.angle[0] -= 6
        if (action - 2) % 3 == 0: #TURN_LEFT
            self.angle[0] += 6
        self.angle[0] = self.angle[0] % 360

    def __call__(self, prev_action, obs, r, done, info):
        self._update_angle(prev_action)
        img = info['brain_info'].visual_observations[0][0]
        speed = info['brain_info'].vector_observations[0]
        absolute_speed = self._rotate_XZ(speed, self.angle[0])
        self.pos[:] += absolute_speed
        obs = {
            "image": img,
            "speed": absolute_speed / 10.,  # (-21.,21.)/10. --> (-2.1,2.1)
            "angle": self.angle / 180 - 1.,  # (0., 360.)/180 -1. --> (-1.,1.)
            "pos": self.pos / 70.,  # (-700.,700.)/70. --> (-10.,10)
            "time":(self.time_limit - self.t) / 250
        }
        self.t += 1
        for p in self.preprocessors:
            obs, r, done, info = p.step(prev_action, obs, r, done, info)

        return super(ExtraObsAdapter, self).__call__(
            prev_action, obs, r, done, info
        )


def make_obs_adapter(**kwargs):
    if kwargs.get('image_only', True):
        return ObservationAdapter(**kwargs)
    else:
        return ExtraObsAdapter(**kwargs)


class Agent(object):

    def _load_train_args(self, folder, file_name='train_args.json'):
        file_path = ospath.join(folder, file_name)
        if ospath.isfile(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return None

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
        self.prev_action = 0

        self.adapt_act = ActionAdapter(
            **self.config.get('action_adapter', {})
        )
        self.adapt_obs = make_obs_adapter(
            device=self.device,
            **self.config.get("observation_adapter",{})
        )

        self.report_config()

    def _check_config(self, config):
        model_path = config['model_path']
        train_args = self._load_train_args(ospath.dirname(model_path))
        obs_adapter_args = config['observation_adapter']
        assert obs_adapter_args['frame_stack'] == train_args['frame_stack'],\
        "frame_stack={} in train_args.json but frame_stack={} in sub_config.yaml".format(
            train_args['frame_stack'], obs_adapter_args['frame_stack']
        )

        oracle_args = train_args.get('real_oracle_args', None)
        if oracle_args:
            train_keys = set(oracle_args.keys())
            conf_keys = set(obs_adapter_args['grid_oracle'])
            keys_match = conf_keys == train_keys
            msg = "GridOracle args are different in config:\n" + \
            "train_args version: {}\n,".format(oracle_args) + \
            " sub_config version: {}".format(obs_adapter_args['grid_oracle'])
            values_match = True
            #for k in train_keys:
            #    oracle_args[k] =

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
        self.adapt_obs.reset(t)
        self.adapt_act.reset()
        self.prev_action = 0

    def _start_episode(self, batch_size=1):

        self.rnn_state = th.zeros(
            batch_size,
            self.model.recurrent_hidden_state_size,
            device=self.device
        )
        self.masks = th.zeros(batch_size,1, device=self.device)
        self.episode_is_running = True
        self.prev_action = 0

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

            obs = self.adapt_obs(self.prev_action, obs, reward, done, info)

            _, action, _, self.rnn_state = self.model.act(
                obs, self.rnn_state,
                self.masks, deterministic=self.greedy
            )

            if self.batch_size == 1:
                action = action.squeeze()

            action = action.tolist()
            self.prev_action = action

            action = self.adapt_act(action)

            if done:
                self.episode_is_running = False

        return action

    def _get_batch_size(self, var):
        if np.isscalar(var) or len(var) == 1:
            return 1
        else:
            return len(var)
