import gym
import numpy as np
from gym import spaces
#import typing
#from typing import Union, Tuple

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
            oracle_reward=2.5/100.,
            # if penatly is true agent is punished for revisiting cells
            # otherwise it is rewarded for visiting new cells
            penalty_mode=True,
            cell_size=(1., 0.5, 1.),
            trace_decay=1., # this means no decay!
            revisit_threshold=0.01,
            exploration_only=False,
            grid_size=None,
    ):
        assert oracle_reward >= 0., \
            "oracle_reward is non-negative! Use `penalty_mode=True` " \
            "if you want to punish an agent for the lack of exploration"

        super(GridOracle, self).__init__()
        self.oracle_r = oracle_reward
        self.cell_size = np.array(cell_size)
        self.penalty_mode=penalty_mode
        self.exploration_only=int(exploration_only)
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


"""
class GridBasedExploration(gym.Wrapper):

    def __init__(
        self, env,
        visiting_r=1./100.,
        grid_size=(31,5,31), # we start at the center of X, and Z, dimensions and at the bottom of Y
        cell_size=(1., 1/2., 1.),
        observe_map=False, #the map is trasposed to shape (Y, X, Z)
        trace_decay=1.,  # this means no decay!
        revisit_threshold = 0.01
    ):
        #assert isinstance(env.observation_space, space.Dict), "This wrapper use obs['pos']!"
        super(GridBasedExploration, self).__init__(env)
        self.grid_size=np.array(grid_size)
        self.cell_size=np.array(cell_size)
        self.visiting_r = visiting_r
        self.observe_map = observe_map
        self._visited = np.zeros(grid_size, dtype=np.float32)
        self.revisit_threshold = revisit_threshold
        self.total_expl_r = 0.
        self.num_visited = 0
        self.trace_decay = trace_decay

        if observe_map:
            spaces = dict(self.observation_space.spaces)
            spaces['visited'] = space.Box(
                0., 1.,
                shape=(grid_size[1], grid_size[0], grid_size[2]),#Y, X, Z
                dtype=np.float32
            )
            self.observation_space = space.Dict(spaces)

    def set_start_coords(self):
        self.start_x = self.grid_size[0] // 2
        self.start_y = 0 # Y is a vertical dimension but we start slightly bellow the ground somehow
        self.start_z = self.grid_size[2] // 2

    def reset(self, **kwargs):
        self._visited[:] = 0
        self.set_start_coords()
        self._visited[self.start_x, self.start_y, self.start_z] = 1.
        self.num_visited = 1
        obs = self.env.reset(**kwargs)
        return self._observation(obs)

    def _observation(self, obs):
        if self.observe_map:
            obs['visited'] = self._visited.copy().transpose(1, 0, 2)
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        if self.trace_decay<1.:
            self._visited *= self.trace_decay
        cell_pos = obs['pos']/self.cell_size
        x, y, z = np.round(cell_pos).astype(np.int32)
        expl_r = self.visit(x,y,z)
        obs = self._observation(obs)
        self._fill_info(obs, expl_r, done, info)

        return obs, r+(expl_r-self.visiting_r), done, info

    def _fill_info(self, obs, r, done, info):
        self.total_expl_r += r
        grid_info = {"n_visited": self.num_visited}
        if self.visiting_r > 0.:
             grid_info["r"] = r
        if done:
            grid_info['episode_r'] = self.total_expl_r
        info['grid_oracle'] = grid_info

    def visit(self, x,y,z):
        x += self.start_x
        y += self.start_y
        z += self.start_z
        r = self.visiting_r if self._visited[x,y,z] < self.revisit_threshold else 0.

        self.num_visited += int(r != 0.)
        self._visited[x,y,z] = 1.
        return r
"""