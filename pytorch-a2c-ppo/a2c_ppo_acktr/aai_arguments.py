import argparse
import datetime
import torch
import os.path as ospath
import json

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    #AAI specific arguments:
    parser.add_argument(
        '--env-path',
        default='aai_resources/env/AnimalAI',
        help='Path to the game file')

    parser.add_argument(
        '--docker-training',
        action='store_true',
        default=False,
        help='Training in docker')

    parser.add_argument(
        '--restart',
        default=None,
        help='Path to model checkpoint')

    parser.add_argument(
        '--config-dir',
        default='aai_resources/default_configs',
        help='Path to a directory with AnimalAI default_configs')

    parser.add_argument(
        '-hl', '--headless',
        action='store_true',
        default=False,
        help='Use headless mod to train UnityEnvironment on server'
    )
    parser.add_argument(
        '--extra-obs', type=str, nargs='*', default=tuple(),
        help="A list of additional observations for agents to receive. "
             "Possible choices are: pos, speed, angle"
    )
    parser.add_argument(
        '-fs', '--frame-stack', type=int, default=2,
        help="Number of image frames to stack into agent's observation, (default: 2)",
    )
    parser.add_argument(
        '--reduced-actions',
        action='store_true',
        default=False,
        help="Removes backward movements from agent's action space. (default: False)"
    )
    #GRID-ORACLE arguments:
    #parser.add_argument(
    #    "--oracle-type", '-ot', default="angles", choices=("3d", "angles"),
    #    help="Which GridOracle you want to use, hint: use angles"
    #)
    parser.add_argument(
        '--oracle-num-angles', '-ona', default=15, type=int,
        help='Number of angle bins in the visitation map. '
             '(default: 15, this means one bin covers 24 degrees)'
    )
    parser.add_argument(
        '--oracle-cell-side', '-ocs', default=2., type=int,
        help='Size of a single grid cell. (default: 2.)'
    )
    parser.add_argument(
        "--oracle-reward", "-or", default=-1./100., type=float,
        help=" If reward > 0 then agents gets this reward when it visits grid cell,"
             " otherwise it is a penalty given to the agent when it stays in the visited"
             " cell for more than one step. (default: -1./100)"
        )


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
    #parser.add_argument(
    #    '--eval-interval',
    #    type=int,
    #    default=None,
    #    help='eval interval, one eval per n updates (default: None)')
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
             '(default: <current-time>)'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='choose your gpu device, if device == -1 then use cpu!')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '-rnn','--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = args.device >= 0 and torch.cuda.is_available()

    if args.restart:
        d = vars(args)
        config_path = ospath.dirname(args.restart)
        with open(config_path + '/train_args.json', 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            d[k] = v

    if not getattr(args, 'experiment_tag', None):
        date = datetime.datetime.now()
        args.experiment_tag = "{0.year}-{0.month}-{0.day}-{0.hour}-{0.minute}".format(date)

    args.summary_dir = ospath.join(args.save_dir, "summaries", args.experiment_tag)

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
