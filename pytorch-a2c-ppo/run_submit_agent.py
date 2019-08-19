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
    import sys
    sys.path.append('submission')
    from agent import Agent
    from a2c_ppo_acktr.aai_config_generator import SingleConfigGenerator, ListSampler
    gen_config = ListSampler.create_from_dir("aai_resources/default_configs/")
    #gen_config = SingleConfigGenerator.from_file("aai_resources/default_configs/1-Food.yaml")
    agent = Agent('submission/data/sub_config.yaml')
    env = create_env()

    #config = gen_config.next_config()
    #print("config name:", config['config_name'])
    #obs = env.reset(config['config'])
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