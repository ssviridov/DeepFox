import numpy as np
import glob
import pathlib
import os.path
from animalai.envs.arena_config import ArenaConfig, RGB, Item
import logging
import copy
from collections import deque, defaultdict
from glob import glob
import fnmatch

class ConfigGenerator(object):
    """
    Basic class for Config generator classes.
    Create an instance of a subclass and pass it into AnimalAiEnv2.
    The instance should switch default_configs at every episode reset
    """
    def next_config(self, *args, **kwargs):
        raise NotImplementedError()

    def shuffle(self):
        raise NotImplementedError()

    def process_results(self, *args, **kwargs):
        """
        a method to process info about agent's results on a previous config.
        For example to decide what config to choose next
        """
        pass

class SingleConfigGenerator(ConfigGenerator):
    """
    Always returns same config. If config wasn't provided, always returns None
    """
    @classmethod
    def from_file(cls, filepath):
        config = ArenaConfig(filepath)
        return cls(config, os.path.basename(filepath))

    def __init__(self, config=None, config_name=None):
        self.config = config
        self.config_name = config_name

    def next_config(self, *args, **kwargs):
        return {"config":self.config, "config_name":self.config_name}

    def shuffle(self):
        pass

class ListSampler(ConfigGenerator):
    """
    Gets a list of ArenaConfigs, shuffles them, and samples them cyclically.
    """
    @classmethod
    def create_from_dir(cls, config_dir):
        config_files = [f for f in pathlib.Path(config_dir).glob("**/*.yaml")]
        configs = []
        config_names = []
        for cf in config_files:
            config = None

            try:
                config = ArenaConfig(cf)

            except Exception as e:
                logging.warning("{} can't load config from {}: {}".format(
                    cls.__name__, cf, e.args
                ))

            if config:
                configs.append(config)
                config_names.append(os.path.relpath(cf, config_dir))


        if len(configs) == 0:
            raise ValueError("There are no aai default_configs in {} directory".format(config_dir))

        return cls(configs, config_names)

    def __init__(self, configs, config_names=None): #, probs):
        """
        :param configs: a list of ArenaConfig objects
        :param config_names: a names of the configs in the config argument
        """
        super(ListSampler, self).__init__()

        if config_names is None:
            config_names = [str(n) for n in range(1, len(configs) + 1)]

        assert len(config_names) == len(configs), 'len(config_name) == len(configs)'

        combined = list(zip(configs,config_names))
        np.random.shuffle(combined)
        self.configs, self.config_names = zip(*combined)

        self.next_id = 0
        #self.probs = np.asarray(probs) if probs else None
        #assert self.probs is None or len(self.probs) == len(self.default_configs), "wrong number of probabilities were given!"

    def shuffle(self):
        combined = list(zip(self.configs, self.config_names))
        np.random.shuffle(combined)
        self.configs, self.config_names = zip(*combined)

    def next_config(self, *args, **kwargs):
        config = self.configs[self.next_id]
        name = self.config_names[self.next_id]
        self.next_id = (self.next_id + 1) % len(self.configs)
        return {'config':config, "config_name":name}


class HierarchicalSampler(ConfigGenerator):

    @classmethod
    def create_from_dir(cls, config_dir):
        import fnmatch
        import os

        n_configs = 0
        root_dict = {}
        for root, dirs, filenames in os.walk(config_dir):
            path = os.path.relpath(root, config_dir)

            curr_dict = root_dict
            if path != '.':
                for d in os.path.normpath(path).split(os.path.sep):
                    curr_dict = curr_dict.setdefault(d, {})

            for fn in fnmatch.filter(filenames, '*.yaml'):
                filename = os.path.join(root, fn)
                config = None
                try:
                    config = ArenaConfig(filename)

                except Exception as e:
                    logging.warning("{} can't load config from {}: {}".format(
                        cls.__name__, filename, e.args
                    ))

                if config:
                    n_configs += 1
                    curr_dict[fn] = config

        if n_configs == 0:
            raise ValueError("There are no aai default_configs in {} directory".format(config_dir))

        return cls(root_dict)

    def __init__(self, names2configs): #, probs):
        """
        :param names2configs: a dictionary that recursively stores filaname -> ArenaConfig mapping
        """
        super(HierarchicalSampler, self).__init__()
        self._configs = names2configs

    def shuffle(self):
        pass

    def next_config(self, *args, **kwargs):
        return self.recursive_choice(self._configs)

    def recursive_choice(self, d, name=""):
        key = np.random.choice(list(d.keys()))
        choice = d[key]
        name = os.path.join(name, key)
        if isinstance(choice, dict):
            return self.recursive_choice(choice, name=name)
        else:
            return {'config':choice, "config_name":name}


def deep_config_update(target, source):
    """Really copies everyhing from source config to target config"""

    for arena_i in source.arenas:
        target.arenas[arena_i] = copy.deepcopy(source.arenas[arena_i])


class ConfigGeneratorWrapper(ConfigGenerator):
    """
    For classes that modify already generated configs.
    For example: change time, add objects, etc.
    """
    def __init__(self, config_generator):
        self.generator = config_generator
        self._tmp_config = ArenaConfig()

    def shuffle(self):
        self.generator.shuffle()

    def next_config(self, *args, **kwargs):
        config_dict = self.generator.next_config(*args, **kwargs)
        deep_config_update(self._tmp_config, source=config_dict['config'])
        config_dict['config']= self._tmp_config
        return self.process_config(config_dict)

    def process_config(self, config_dict):
        raise NotImplementedError


class FixedTimeGenerator(ConfigGeneratorWrapper):
    """
    All configs will have the same time limit.
    I use it in aai_interact.py
    """
    def __init__(self, config_generator, time):
        super(FixedTimeGenerator, self).__init__(config_generator)
        self.time = time

    def process_config(self, config_dict):
        config_dict['config'].arenas[0].t = self.time
        return config_dict


class RandomizedGenerator(ConfigGeneratorWrapper):
    """
    A generator that randomizes objects properties:
    changes color, sets blackouts, replaces an object with similar one.

    !NOT DONE YET!
    """

    PERMUTABLE_OBJECTS=(
        ["WallTransparent", "Wall"],
        ["CardBox1", "CardBox2"],
        ["CylinderTunnel", "CylinderTunnelTransparent"],
        ["UObject", "LObject", "LObject2"]
    )

    PERMUTABLE_COLORS = (
        [(255, 0, 255), (153, 153, 153), (-1., -1, -1), (0, 255, 0), (255, 215, 0)]
    )

    PAINTABLE_OBJECTS = {'Wall', 'CylinderTunnel', "Ramp"}

    BLACKOUTS = [[-10], [-20], [-30], [-40]]

    @staticmethod
    def create(config_generator, blackout_prob, object_change_prob, color_change_prob):
        if blackout_prob>0. or object_change_prob>0. or color_change_prob>0.:
            return RandomizedGenerator(config_generator, blackout_prob, object_change_prob, color_change_prob)
        else: #if you change nothing why bother?
            return config_generator

    def __init__(self, config_generator, blackout_prob=0.1, object_change_prob=0.2, color_change_prob=0.2):
        super(RandomizedGenerator, self).__init__(config_generator)
        self.blackout_prob=blackout_prob
        self.object_change_prob = object_change_prob
        self.color_change_prob = color_change_prob

    def process_config(self, config_dict):
        config = config_dict['config']
        #print('Randomizing[ocp={:0.2f}, ccp={:0.2f}] {}'.format(self.object_change_prob, self.color_change_prob, config_dict['config_name']))


        for k, arena in config.arenas.items():
            if not len(arena.blackouts) and arena.t <= 500:
                add_blackout = self.blackout_prob > np.random.random()

                if add_blackout:
                    blackout_id =np.random.choice(len(self.BLACKOUTS))
                    arena.blackouts = self.BLACKOUTS[blackout_id]
                    arena.t = arena.t*2

            for it_id, item in enumerate(arena.items):
                change_obj = self.object_change_prob > np.random.random()
                #print('Try to change {}: {}'.format(item.name, change_obj))
                if change_obj:
                    for group in self.PERMUTABLE_OBJECTS:
                        if item.name in group:
                            new_name = np.random.choice(group)
                            #print('Changing {} with {}'.format(item.name, new_name))
                            if new_name != item.name:
                                arena.items[it_id] = Item(new_name, item.positions, item.rotations, item.sizes, item.colors)
                                item = arena.items[it_id]
                                #print('Object {} was changed to {}'.format(item.name, new_name))
                            break
                if item.name in self.PAINTABLE_OBJECTS:

                    n_colors = len(item.colors) #colors specified
                    n_items = max(n_colors,len(item.positions), len(item.sizes)) #number different items
                    for i in range(n_items-n_colors): item.colors.append(RGB(-1,-1,-1))

                    for c_id, rgb in enumerate(item.colors):
                        if self.color_change_prob > np.random.random():
                            color = rgb.r, rgb.g, rgb.b
                            if color in self.PERMUTABLE_COLORS:
                                new_color_id = np.random.choice(len(self.PERMUTABLE_COLORS))
                                new_color = self.PERMUTABLE_COLORS[new_color_id]
                                item.colors[c_id] = RGB(*new_color)
                                #print('{}#{}: changed RGB{} to RGB{}'.format(item.name, c_id, color, new_color))

        return config_dict


class ConfigMerger(ConfigGenerator):
    """
    Takes list of config generators and list with their respective probabilities.
    """

    def __init__(self, config_generators, probs):
        assert sum(probs) == 1, "Probabilities must sum to 1"
        self.config_generators = config_generators
        self.probs = probs
        self.prev_chosen_gen = None

    def next_config(self, *args, **kwargs):
        if self.prev_chosen_gen:
            self.prev_chosen_gen.process_results(kwargs.pop('config_results'))

        chosen_gen = np.random.choice(self.config_generators, p=self.probs)
        self.prev_chosen_gen = chosen_gen
        return chosen_gen.next_config(*args, **kwargs)

    def shuffle(self):
        for cg in self.config_generators:
            cg.shuffle()



class Curriculum(ConfigGenerator):

    @classmethod
    def create_from_dir(cls, config_dir, **kwargs):
        dir_dict = defaultdict(dict)
        dirs = glob(config_dir+'/*/')
        for name in dirs:
            config_files = [f for f in pathlib.Path(name).glob("**/*.yaml")]
            file_dict = dict()
            for file in config_files:
                config = ArenaConfig(file)
                file_dict[file] = config
            dir_dict[name] = file_dict

        return cls(dir_dict, **kwargs)

    def __init__(
            self,
            folders2configs,
            thresholds=(0.3, 0.6),
            min_update_period=100,
            history_len=70,
    ):
        self._configs = folders2configs
        self._folders = sorted(list(folders2configs.keys()))
        self.name = os.path.basename(os.path.commonpath(self._folders))
        print('Curriculum on', self.name)

        self._level = 0
        self._probs = [0.] * len(self._folders)
        self._update_probs(0.)

        self.lower_threshold = thresholds[0]
        self.upper_threshold = thresholds[1]

        self.min_update_period = min_update_period
        # history >= min_update_period:
        self.history_len = max(history_len if history_len else 0, min_update_period)
        self.successes = deque(maxlen=max(min_update_period,history_len))
        self.n_episodes = 0
        self.prev_config = None

    def shuffle(self):
        pass

    def process_results(self, config_results):
        if config_results['name'] != self.prev_config:
            raise ValueError("Generator expects results from config it generated previously")

        self.successes.append(config_results['success'])
        #print("n-ep:", self.n_episodes, 'success:', len(self.successes),"lvl:", self._folders[self._level] )

        if len(self.successes) >= self.min_update_period:
            quality = np.mean(self.successes)
            new_level = self._level

            if quality >= self.upper_threshold:
                new_level = min(self._level+1, len(self._folders)-1)
                #print('!LEVEL UP! to {}-level-{}'.format(self.name, new_level+1))

            elif quality < self.lower_threshold:
                new_level = max(self._level-1, 0)
                #print('LEVEL DOWN to {}-level-{}'.format(self.name, new_level+1))

            if new_level != self._level:
                self._level = new_level
                self._update_probs(quality)
                self.successes.clear()

        if self.n_episodes % 50 == 0:
            quality = np.mean(self.successes) if self.successes else quality
            print("{}-level-{}".format(
                self.name, self._level+1),
                "N[e]:", self.n_episodes,
                "N[s]:", len(self.successes),
                'quality:', quality
            )

    def next_config(self, *args, **kwargs):
        level_folder = np.random.choice(self._folders, p=self._probs)
        cfg_name = np.random.choice(list(self._configs[level_folder].keys()))
        self.prev_config = os.path.dirname(cfg_name)
        self.n_episodes += 1
        return {'config':self._configs[level_folder][cfg_name], "config_name":self.prev_config}


    def _update_probs(self, quality):
        self._probs[self._level] = 0.7
        new_probs = np.array([0.7 * (0.2 ** abs(i - self._level)) for i in range(len(self._probs))])
        self._probs = new_probs / sum(new_probs)