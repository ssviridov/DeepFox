import gym
import numpy as np
from gym import spaces
#import typing
#from typing import Union, Tuple
from .object_classifier import StateClassifier
import torch as th

class Preprocessor(object):

    def init(self, env):
        """
        if you need information from environment to
        build your preprocessor do this here.
        This method should also return updated(if required)
        observation space, action space, reward range and metadata
        :returns: updated action and observation spaces
        """
        return (env.observation_space,
                env.action_space,
                env.reward_range,
                env.metadata)

    def reset(self, obs):
        raise NotImplementedError()

    def step(self,  act, obs, r, done, info):
        raise NotImplementedError()

    def wrap_env(self, env):
        return PreprocessingWrapper(env, self)


class PreprocessingWrapper(gym.Wrapper):

    def __init__(self, env, preprocessor):
        if not hasattr(env, 'reward_range'):
            env.reward_range = None
        if not hasattr(env, 'metadata'):
            env.metadata = None

        super(PreprocessingWrapper, self).__init__(env)

        obs_sp, act_sp, r_range, metadata = preprocessor.init(env)
        self.action_space = act_sp
        self.observation_space = obs_sp
        self.reward_range = r_range
        self.metadata = metadata

        self.preprocessor = preprocessor

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.preprocessor.reset(obs, **kwargs)
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs, r, done, info = self.preprocessor.step(action, obs, r, done, info)
        return obs, r, done, info

    @th.no_grad()
    def step_async(self, actions):
        #VecEnv and it's subclasses may use step_async and step_wait instead of step
        self.prev_action = actions
        self.env.step_async(actions)

    @th.no_grad()
    def step_wait(self):
        obs, r, done, info = self.env.step_wait()
        obs, r, done, info = self.preprocessor.step(self.prev_action, obs, r, done, info)
        return obs, r, done, info


class GridOracle(Preprocessor):
    """
    Splits arena into 3D cells with the center at agent's start location.
    To do this GridOracle should have access to the "pos" and "speed"
    observations, and agent's (flattened) actions.
    GridOracle can reward an agent for visiting new cells during the episode,
    or can punish him for staying in the same place to long.
    GridOracle also adds a map of 'visited' cells to the observation dicts.
    """
    GRID_SIZE=(31,7, 31)
    CELL_SIZE=(1.,0.5, 1.)

    @staticmethod
    def compute_grid_size(cell_size, keep_odd=False):
        """
        For cell size [X=1,Y=0.5, Z=1] the will need a grid no bigger than
        [X=31, Y=7, X=31](probably an overestimate for X and Z).
        For different cell sizes this method returns proportionally scaled
        size of the grid.
        You also better go and run aai_interact.py to empirically estimate
        your new grid
        """
        if GridOracle.CELL_SIZE == cell_size:
            return GridOracle.GRID_SIZE

        cell_ratio = np.array(cell_size)/np.array(GridOracle.CELL_SIZE)
        grid_size = np.array(GridOracle.GRID_SIZE)/cell_ratio
        X, Y, Z = np.ceil(grid_size).astype(int)

        if keep_odd:
            X += int(X % 2 == 0)
            Z += int(Z % 2 == 0)

        return X, Y, Z

    def __init__(
            self,
            oracle_reward=-2.5/100.,
            # if reward <= 0. agent is punished for revisiting cells
            # otherwise it is rewarded for visiting new cells
            #penalty_mode=True,
            cell_size=(1., 0.5, 1.),
            trace_decay=1., # this means no decay!
            revisit_threshold=0.01,
            exploration_only=False,
            grid_size=None,
    ):
        #assert oracle_reward >= 0., \
        #    "oracle_reward is non-negative! Use `penalty_mode=True` " \
        #    "if you want to punish an agent for the lack of exploration"

        super(GridOracle, self).__init__()
        self.oracle_r = np.abs(oracle_reward)
        self.cell_size = np.array(cell_size)
        self.penalty_mode=(oracle_reward <= 0.0)
        self.exploration_only=exploration_only
        self.revisit_threshold = revisit_threshold
        self.trace_decay = trace_decay

        if grid_size is None:
            self.grid_size = self.compute_grid_size(cell_size)
        else:
            self.grid_size = grid_size
        self._visited = np.zeros(shape=self.grid_size, dtype=np.float32)

        self.total_expl_r = 0.
        self.num_visited = 0
        self._set_start_coords()

    def init(self, env):

        space = dict(env.observation_space.spaces)
        # we return observation as Y, X, Z
        shape = (self.grid_size[1], self.grid_size[0], self.grid_size[2])
        space['visited'] = spaces.Box(
            0., 1., shape=shape, dtype=np.float32
        )
        obs_space = spaces.Dict(space)

        return (obs_space,
                env.action_space,
                env.reward_range,
                env.metadata)

    def reset(self, obs):
        self._visited[:] = 0
        self._visited[self.start_x, self.start_y, self.start_z] = 1.
        self.num_visited = 1
        return self._observation(obs)

    def step(self, act, obs, r, done, info):
        if self.trace_decay<1.:
            self._visited *= self.trace_decay
        cell_pos = obs['pos']/self.cell_size
        x, y, z = np.round(cell_pos).astype(np.int32)
        expl_r = self._visit(x, y, z)
        obs = self._observation(obs)
        self._fill_info(obs, expl_r, done, info)

        if self.exploration_only:
            r = expl_r
        else:
            r = r + expl_r

        return obs, r, done, info

    def _set_start_coords(self):
        self.start_x = self.grid_size[0] // 2
        self.start_y = 0 # Y is a vertical dimension but we start slightly bellow the ground somehow
        self.start_z = self.grid_size[2] // 2

    def _observation(self, obs):
        obs['visited'] = self._visited.copy().transpose(1, 0, 2)
        return obs

    def _fill_info(self, obs, r, done, info):
        self.total_expl_r += r
        grid_info = {"n_visited": self.num_visited}
        if self.oracle_r > 0.:
             grid_info["r"] = r
        if done:
            grid_info['episode_r'] = self.total_expl_r
        info['grid_oracle'] = grid_info

    def _visit(self, x, y, z):
        x += self.start_x
        y += self.start_y
        z += self.start_z
        r = 0.

        if self._visited[x, y, z] < self.revisit_threshold:
            r += self.oracle_r #reward for visiting new state

        if self.penalty_mode:#(+oracle_r or 0.) becomes a (0. or -oracle_r)
            r -= self.oracle_r

        self.num_visited += (self._visited[x,y,z] == 0.)
        self._visited[x,y,z] = 1.
        return r


class GridOracleWithAngles(Preprocessor):
    """
        Splits arena into 2D cells(and angle boxes as first dimension) with the
        center at agent's start location. To do this GridOracle should have
        access to the ("pos", "speed", "angle") observations, and agent's
        (flattened) actions. GridOracle can reward an agent for visiting new
        cells during the episode, or can punish him for staying in the same
        place to long. GridOracle also adds a map of 'visited' cells to
        the observation dicts.
        """
    GRID_SIDE = 31
    CELL_SIDE = 1.
    ANGLE_RANGE = 2.
    ANGLE_MIN = -1.

    @staticmethod
    def compute_grid_size(cell_side, num_angles, keep_odd=False):
        """
        For cell side = 1. the will need a grid side no bigger than
         31 (probably an overestimate for number of cells).
        For different cell sizes this method returns proportionally scaled
        size of the grid.
        You also better go and run aai_interact.py to empirically estimate
        your new grid
        """
        if GridOracleWithAngles.CELL_SIDE == cell_side:
            return (
                num_angles,
                GridOracleWithAngles.GRID_SIDE,
                GridOracleWithAngles.GRID_SIDE
            )

        cell_ratio = np.array(cell_side) / np.array(GridOracleWithAngles.CELL_SIDE)
        grid_side = np.array(GridOracleWithAngles.GRID_SIDE) / cell_ratio
        X = int(np.ceil(grid_side))

        if keep_odd:
            X += int(X % 2 == 0)

        return (num_angles, X, X)

    def __init__(
            self,
            oracle_reward=-1./100.,
            # if reward < 0. agent is punished for revisiting cells
            # otherwise it is rewarded for visiting new cells
            #penalty_mode=True,
            cell_side=1.,
            num_angles=6,
            trace_decay=1.,  # this means no decay!
            revisit_threshold=0.01,
            exploration_only=False,
            grid_side=None,
            oracle_reward_final=None,
            total_steps=None
    ):
        #assert oracle_reward >= 0., \
        #    "oracle_reward is non-negative! Use `penalty_mode=True` " \
        #    "if you want to punish an agent for the lack of exploration"

        super(GridOracleWithAngles, self).__init__()
        self.oracle_r = np.abs(oracle_reward)
        self.cell_side = cell_side
        self.angle_box = self.ANGLE_RANGE/num_angles
        self.curr_step = 0
        self.oracle_r_final = oracle_reward_final

        if self.oracle_r_final is not None:
            assert total_steps is not None
            self.total_steps = total_steps
            self.delta_r = (self.oracle_r - self.oracle_r_final)/self.total_steps

        self.penalty_mode = (oracle_reward <= 0.0)
        self.exploration_only = exploration_only
        self.revisit_threshold = revisit_threshold
        self.trace_decay = trace_decay

        if grid_side is None:
            self.grid_size = self.compute_grid_size(cell_side, num_angles)
        else:
            self.grid_size = (num_angles, grid_side, grid_side)
        self._visited = np.zeros(shape=self.grid_size, dtype=np.float32)

        self.total_expl_r = 0.
        self.num_visited = 0
        self._set_start_coords()

    def init(self, env):
        angle_obs= env.observation_space['angle']
        assert angle_obs.low == -1. and angle_obs.high == 1., \
            "GridOracleWithAngles expects angle to be in (-1., 1.) range, i.e (real_angle/180 - 1.)"

        space = dict(env.observation_space.spaces)
        # we return observation as Y, X, Z
        space['visited'] = spaces.Box(
            0., 1., shape=self.grid_size, dtype=np.float32
        )
        obs_space = spaces.Dict(space)

        return (obs_space,
                env.action_space,
                env.reward_range,
                env.metadata)

    def reset(self, obs):
        #print('GRID ORACLE IS RESET!')
        self._visited[:] = 0
        self.num_visited = 0
        self.total_expl_r = 0.

        #coords = self._cell_coords(obs['pos'], obs['angle'])
        #expl_r = self._visit(*coords)

        return self._observation(obs)

    def step(self, act, obs, r, done, info):
        #if done:  print('GRID ORACLE HAS SEEN DONE!')
        self._discount_visited()

        coords = self._cell_coords(obs['pos'], obs['angle'])
        #print("Angle: {:.1f}, Box_id: {}".format((obs['angle'][0]+1.)*180, coords[0]))
        expl_r = self._visit(*coords)

        obs = self._observation(obs)
        self._fill_info(obs, expl_r, done, info)

        if self.exploration_only:
            r = expl_r
        else:
            r = r + expl_r

        self.curr_step += 1
        #print('visited != 0: ', (self._visited > 0.).sum(),
        #      'num_visited:', self.num_visited,
        #      "visited now: ", (self._visited > 0.5).sum(),)

        return obs, r, done, info

    def _discount_visited(self):
        #call this function before marking current cell in the grid
        if self.trace_decay < 1.: #this option gradually discounts old states
            self._visited *= self.trace_decay
        else: #all previosly visited locations are equal to 0.5!
            self._visited = np.minimum(self._visited, 0.5)

    def _cell_coords(self, pos, angle):
        x, y, z = pos #y is a vertical dimension and we ignore it!
        num_angles = self.grid_size[0]
        angle_id = int(np.round(angle / self.angle_box))
        angle_id = angle_id % num_angles
        x_id = int(np.round(x/self.cell_side))
        z_id = int(np.round(z/self.cell_side))

        return (angle_id, x_id, z_id)

    def _set_start_coords(self):
        #num_angles, _  = self.grid_size.shape
        self.start_angle = 0
        self.start_x = self.grid_size[1] // 2
        self.start_z = self.grid_size[2] // 2

    def _observation(self, obs):
        obs['visited'] = self._visited.copy()
        return obs

    def _fill_info(self, obs, r, done, info):
        self.total_expl_r += r
        grid_info = {"n_visited":self.num_visited}
        if self.oracle_r > 0.:
            grid_info["r"] = r
        if done:
            grid_info['episode_r'] = self.total_expl_r
        info['grid_oracle'] = grid_info

    def _visit(self, angle, x, z):
        if self.oracle_r_final is not None:
            if self.curr_step > self.total_steps:
                curr_oracle_r = self.oracle_r_final
            else:
                curr_oracle_r = self.oracle_r - self.delta_r*self.curr_step

        else:
            curr_oracle_r = self.oracle_r

        angle += self.start_angle
        x += self.start_x
        z += self.start_z
        r = 0.

        if self._visited[angle, x, z] < self.revisit_threshold:
            r += curr_oracle_r  # reward for visiting new state
            #oracle_r = self.oracle_r*(self.grid_size[0] - sum(self._visited[:, x, z] != 0.) / self.grid_size[0])

        if self.penalty_mode:  # (+oracle_r or 0.) becomes a (0. or -oracle_r)
            r -= curr_oracle_r

        self.num_visited += (self._visited[angle, x, z] <= 0.)
        self._visited[angle, x, z] = 1.
        return r


class MetaObs(Preprocessor):
    """
    Adds extra observations typically used in meta-learning settings
    r_{t-1}, a_{t-1} in the observations dict under keys `r_prev`, `a_prev`.
    """

    def __init__(self, one_hot_actions=True, num_actions=None):
        self.one_hot_actions = one_hot_actions
        if num_actions or one_hot_actions == False:
            self.action_dim = num_actions if self.one_hot_actions else 6
        self.r_prev = None

    def init(self, env):
        assert isinstance(env.observation_space, spaces.Dict), "Works only with dict obs space!"
        #update obs space with new observations:
        space = dict(env.observation_space.spaces)

        space['r_prev'] = spaces.Box(-6., 6., shape=(1,), dtype=np.float32)
        if not hasattr(self, "action_dim"):
            self.action_dim = env.action_space.n if self.one_hot_actions else 6
        space['a_prev'] = spaces.Box(0., 1., shape=(self.action_dim,), dtype=np.float32)

        obs_space = spaces.Dict(space)

        return (obs_space,
                env.action_space,
                env.reward_range,
                env.metadata)

    def reset(self, obs):
        self.r_prev = np.zeros((1,), dtype=np.float32)
        obs['a_prev'] = np.zeros((self.action_dim,), dtype=np.float32)
        obs['r_prev'] = self.r_prev
        return obs

    def step(self,  act, obs, r, done, info):
        obs['a_prev'] = self._get_action_repr(act)
        obs['r_prev'] = self.r_prev.copy()
        self.r_prev[0] = r
        return obs, r, done, info

    def _get_action_repr(self, action):
        if not self.one_hot_actions:
            first_id, second_id = action//3, action % 3
            action_repr = np.zeros((self.action_dim,), dtype=np.float32)
            action_repr[first_id] = 1.
            action_repr[second_id+3] = 1.
            return action_repr
        else:
            action_repr = np.zeros((self.action_dim,), dtype=np.float32)
            action_repr[action] = 1.
            return action_repr

class ObjectClassifier(Preprocessor):
    """
    Multi-label object classifier.
    After ObjectClassifier an observation dict returned by environment will store
    probabilities of each object type present on the image.
    In case you are using frame stack it looks at the last 3 channels
     in the image tensor (the newest frame in the stack).
    """
    def __init__(
        self,
        model_path='aai_resources/classifier/best_model.pkl',
        threshold=None,
        device='cpu'
    ):
        self.threshold = threshold
        self.clf = StateClassifier()
        self.clf.load_state_dict(th.load(model_path, map_location=device))
        self.clf = self.clf.to(device)
        self.clf.eval()

        self.num_labels = self.clf.NUM_LABELS
        self.labels = self.clf.LABELS
        self.device = device

    def init(self, env):
        assert isinstance(env.observation_space, spaces.Dict), "Works only with dict obs space!"
        #update obs space with new observations:
        space = dict(env.observation_space.spaces)

        shape = (self.clf.NUM_LABELS,)
        space['objects'] = spaces.Box(0., 1., shape=shape, dtype=np.float32)

        obs_space = spaces.Dict(space)

        return (obs_space,
                env.action_space,
                env.reward_range,
                env.metadata)

    def reset(self, obs):
        # during evaluation first reset is empty for preprocessors:
        if 'image' in obs:
            return self.add_objects_to_obs(obs)

        return obs

    def step(self,  act, obs, r, done, info):
        obs = self.add_objects_to_obs(obs)
        return obs, r, done, info

    @th.no_grad()
    def add_objects_to_obs(self, obs):
        img = obs['image']

        *batch_size, C, H, W = img.shape
        if not batch_size:
            img = th.as_tensor(img, device=self.device).unsqueeze(0)
        else:
            img = img.to(self.device)

        results = th.sigmoid(self.clf(img[:,-3:]))

        if self.threshold:
            results = (results > self.threshold).to(th.float32)

        if not batch_size:
            results = results.squeeze(0).cpu().numpy()
        else:
            results = results.to(obs['image'].device)

        obs['objects'] = results
        return obs