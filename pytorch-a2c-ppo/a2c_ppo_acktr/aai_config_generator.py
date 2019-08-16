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
        return cls(config, os.path.basename(filepath))

    def __init__(self, config=None, config_name=None):
        self.config = config
        self.config_name = config_name

    def next_config(self, *args, **kwargs):
        return {"config":self.config, "config_name":self.config_name}


class ListSampler(ConfigGenerator):
    """
    Gets a list of ArenaConfigs, shuffles them, and samples them cyclically.
    """
    @classmethod
    def create_from_dir(cls, config_dir):
        pattern = os.path.join(config_dir, "*.yaml")
        config_files = glob.glob(pattern)
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
                config_names.append(os.path.basename(cf))


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

        np.random.shuffle(configs)
        self.next_id = 0
        #self.probs = np.asarray(probs) if probs else None
        #assert self.probs is None or len(self.probs) == len(self.default_configs), "wrong number of probabilities were given!"

    def next_config(self):
        config = self.configs[self.next_id]
        name = self.config_names[self.next_id]
        self.next_id = (self.next_id + 1) % len(self.configs)
        return {'config':config, "config_name":name}