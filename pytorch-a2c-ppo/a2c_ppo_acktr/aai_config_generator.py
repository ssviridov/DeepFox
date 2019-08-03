import numpy as np
import glob
import os.path
from animalai.envs.arena_config import ArenaConfig
import logging


class ConfigGenerator(object):
    """
    Basic class for Config generator classes.
    Create an instance of a subclass and pass it into AnimalAiEnv2.
    The instance should switch default_configs at every episode reset
    """
    def next_config(self, *args, **kwargs):
        raise NotImplementedError()


class SingleConfigGenerator(ConfigGenerator):
    """
    Always returns same config. If config wasn't provided, always returns None
    """
    @classmethod
    def from_file(cls, filepath):
        config = ArenaConfig(filepath)
        return cls(config)

    def __init__(self, config=None):
        self.config = config

    def next_config(self, *args, **kwargs):
        return self.config


class ListSampler(ConfigGenerator):
    """
    Gets a list of ArenaConfigs, shuffles them, and samples them cyclically.
    """
    @classmethod
    def create_from_dir(cls, config_dir):
        pattern = os.path.join(config_dir, "*.yaml")
        config_files = glob.glob(pattern)
        configs = []
        for cf in config_files:
            try:
                config = ArenaConfig(cf)
                configs.append(config)
            except Exception as e:
                logging.warn("{} can't load config from {}: {}".format(
                    cls.__name__, cf, e.args
                ))

        if len(configs) == 0:
            raise ValueError("There are no aai default_configs in {} directory".format(config_dir))

        return cls(configs)

    def __init__(self, configs): #, probs):
        super(ListSampler, self).__init__()
        self.configs = configs[:]
        np.random.shuffle(self.configs)
        self.next_id = 0
        #self.probs = np.asarray(probs) if probs else None
        #assert self.probs is None or len(self.probs) == len(self.default_configs), "wrong number of probabilities were given!"

    def next_config(self):
        config = self.configs[self.next_id]
        self.next_id = (self.next_id + 1) % len(self.configs)
        return config