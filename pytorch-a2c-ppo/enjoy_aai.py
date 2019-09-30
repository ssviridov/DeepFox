import argparse
import os
# workaround to unpickle olf model files
import sys
import time
import json

from itertools import count
import numpy as np
import torch

from a2c_ppo_acktr.aai_wrapper import make_vec_envs_aai
from a2c_ppo_acktr.aai_config_generator import ListSampler

sys.path.append('a2c_ppo_acktr')

def load_args(folder, file_name='train_args.json'):
    file_path = os.path.join(folder, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return None

def get_config_name(venv_aai):
    name = env.venv.envs[0].unwrapped.config_name[:-5]#no .yaml suffix
    if name[-1].isdigit():
        return name[:-1] #different instances of one env type have a digit at the end
    return name

parser = argparse.ArgumentParser(description='RL')

parser.add_argument(type=str, dest="model_path",
    help='path to a pretrained model')

parser.add_argument(
    '--seed', type=int, default=None, help='random seed (default: None)')

parser.add_argument(
    '-d', '--delay', type=float, default=0.05, help='Slows down the demonstration speed')

parser.add_argument(
    '--env-path',
    default='aai_resources/env/AnimalAI',
    help='Path to the game file')

parser.add_argument(
    '--config-dir',
    default='aai_resources/default_configs',
    help='Path to a directory with AnimalAI default_configs')

parser.add_argument(
    '-n', '--num-episodes',
    type=int, default=1,
    help='Number of episodes to play!'
)

parser.add_argument(
    '-c', '--cuda', action="store_true", help="Load model and play on gpu"
)

parser.add_argument(
    '--det',
    action='store_true',
    help='whether to use a deterministic policy')

args = parser.parse_args()

if args.seed is None:
    args.seed = np.random.randint(1000)

device = torch.device("cuda:0" if args.cuda else "cpu")
gen_config = ListSampler.create_from_dir(args.config_dir)
#gen_config = SingleConfigGenerator.from_file("aai_resources/default_configs/allObjectsRandom1.yaml")

train_args = load_args(os.path.dirname(args.model_path))

if(train_args):
    image_only = len(train_args.get('extra_obs',[])) == 0
    oracle_kwargs = dict(
        oracle_type=train_args.get("oracle_type","angles"),
        oracle_reward=train_args.get("oracle_rewards", 1./100.)
    )
else:
    image_only = True
    oracle_kwargs = dict()

env = make_vec_envs_aai(
    args.env_path,
    gen_config,
    args.seed,
    1,
    device,
    num_frame_stack=train_args.get('frame_stack', 1),
    headless=False,
    grid_oracle_kwargs=oracle_kwargs,
    image_only=image_only
)

# We need to use the same statistics for normalization as used in training
data = torch.load(args.model_path, map_location=device)
actor_critic = data['model'] if isinstance(data, dict) else data[0]

actor_critic = actor_critic.to(device)

episode_rewards = []
episode_success = []
episode_steps = []

configs2reward={}
configs2success={}

for episode in range(args.num_episodes):
    rnn_state = torch.zeros(1, actor_critic.internal_state_shape).to(device)

    masks = torch.zeros(1, 1).to(device)
    obs = env.reset()

    print("Episode#{}".format(episode+1))

    curr_config = get_config_name(env)
    configs2reward.setdefault(curr_config, [])
    configs2success.setdefault(curr_config, [])

    print("CONFIG: {}".format(curr_config))
    total_r = 0.

    for t in count(1):
        with torch.no_grad():
            value, action, _, rnn_state = actor_critic.act(
                obs, rnn_state, masks, deterministic=args.det)

        # Obs reward and next obs
        obs, reward, done, info = env.step(action)
        total_r += reward.item()
        time.sleep(args.delay)
        if done:

            break
        #masks.fill_(0.0 if done else 1.0)
    episode_rewards.append(info[0]['episode_reward'])
    episode_success.append(info[0]['episode_success'])
    episode_steps.append(t)

    configs2reward[curr_config].append(info[0]['episode_reward'])
    configs2success[curr_config].append(info[0]['episode_success'])

    print('total_r={:0.2f}, num_steps={}\n'.format(total_r, t))

print('Played {} episodes total:'.format(args.num_episodes))
print('Mean R: {:0.2f}'.format(np.mean(episode_rewards)))
print("Median R: {:0.2f}".format(np.median(episode_rewards)))
print('Mean success: {:0.2f}'.format(np.mean(episode_success)))
print("Mean Steps: {:0.1f}".format(np.mean(episode_steps)))
print()
for k in sorted(configs2reward.keys()):
    v = configs2reward[k]
    print("{}: {:0.2f} avr reward".format(k, np.mean(v)))