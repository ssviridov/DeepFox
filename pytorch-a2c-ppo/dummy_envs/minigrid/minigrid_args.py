import argparse
import datetime
import os.path as ospath
import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    #AAI specific arguments:
    parser.add_argument('--game', type=str, default="MiniGrid-MemoryS7-v0")
    parser.add_argument(
        '-fs', '--frame-stack', type=int, default=1,
        help="Number of image frames to stack into agent's observation, (default: 2)",
    )
    parser.add_argument(
        '-hs', '--hidden-size', default=64, type=int, help='Size of hidden layers')
    #PPO/A2C arguments:
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=None, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')

    parser.add_argument(
        '-sd', '--save-dir',
        default='./trained_models/',
        help='Directory to save different experiments and common summaries (default: ./trained_models/)')
    parser.add_argument(
        '-et', '--experiment-tag',
        default=None,
        help='tag of the current experiment. '
             'It affect name of the written summaries, and path to saved weights. '
             '(default: <current-time>)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '-pol', '--policy',
        choices=('rnn', 'ff', 'mha', 'tc', 'cached_tc', 'cached_mha'),
        default="ff",
        help='Choose policy: feedforward, recurrent, or based on multihead attention!'
    )
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not getattr(args, 'experiment_tag', None):
        date = datetime.datetime.now()
        args.experiment_tag = "{0.year}-{0.month}-{0.day}-{0.hour}-{0.minute}".format(date)

    args.summary_dir = ospath.join(args.save_dir, "summaries", args.experiment_tag)

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.policy == "rnn":
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
