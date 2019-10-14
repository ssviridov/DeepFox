from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig

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

def get_next_config(gen_config):
    config_dict = gen_config.next_config()
    name = config_dict['config_name'][:-5]  # no .yaml suffix
    config = config_dict['config']
    if name[-1].isdigit():
        name = name[:-1]  # different instances of one env type have a digit at the end
    return config, name

def recursive_fill(stats, info):
    for k, v in info.items():
        if isinstance(v, dict):
            k_info = v
            k_stats = stats.setdefault(k, {})
            recursive_fill(k_stats, k_info)
        else:
            stats.setdefault(k, []).append(v)

if __name__ == "__main__":
    import random as rnd
    import itertools as it
    import sys
    from collections import defaultdict

    sys.path.append('submission')
    from agent import Agent
    from a2c_ppo_acktr.aai_config_generator import ListSampler, SingleConfigGenerator
    #gen_config = ListSampler.create_from_dir("aai_resources/default_configs/")
    gen_config = SingleConfigGenerator.from_file("aai_resources/default_configs/1-Food.yaml")
    agent = Agent('submission/data/pretrained/gpu2-default2/sub_config.yaml')
    env = create_env(3)

    #config = gen_config.next_config()
    #print("config name:", config['config_name'])
    #obs = env.reset(config['config'])
    #obs = env.reset()
    stats = {}

    print('Running 5 episodes')
    for k in range(5):
        cumulated_reward = 0
        config, name = get_next_config(gen_config)
        print('Episode {} starting: {}'.format(k, name))
        env.reset(config)
        agent.reset(t=config.arenas[0].t)
        try:
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

        recursive_fill(stats, info)

        print(
            'Episode {0} completed, reward {1:0.2f}, num_steps {2}'.format(
                k, cumulated_reward, step
        ))
        #print('R_stat: {.2f}, num_steps: {}, success: {}'.format(
        #   stats['episode_reward'][-1],
        #   stats['episode_len'][-1],
        #   stats['episode_success'][-1],
        #))


    print('SUCCESS')