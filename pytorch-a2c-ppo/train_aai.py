import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.aai_arguments import get_args
from a2c_ppo_acktr.aai_wrapper import make_vec_envs_aai
from a2c_ppo_acktr.aai_models import AAIPolicy, Policy

from a2c_ppo_acktr.aai_storage import create_storage

from evaluate_aai import evaluate
from a2c_ppo_acktr.aai_config_generator import ListSampler, SingleConfigGenerator
from datetime import datetime
import json
from tensorboardX import SummaryWriter


#def curr_day():
#    return datetime.now().strftime("%y_%m_%d")


def ensure_dir(file_path):
    """
    Checks if the containing directories exist,
    and if not, creates them.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class DummySaver(object):

    def __init__(self, args):
        super(DummySaver, self).__init__()
        self.best_quality = float('-inf')
        self.args = args
        self.save_every_updates = args.save_interval
        self.save_subdir = self._build_save_path(args)
        self.save_args()
        self.model_path = os.path.join(self.save_subdir, "under-{}0M-steps.pt")
        self.best_model_path = os.path.join(self.save_subdir, "best.pt")

    def save_model(self, num_updates, total_step, quality, model, optim=None):
        if num_updates % self.save_every_updates == 0 or num_updates == (self.args.total_updates-1):
            # 9950k goes into under-10M-steps.pt
            # 10200k goes into under-20-steps.pt
            lt_steps = str((total_step // 10000000) + 1)
            save_path = self.model_path.format(lt_steps)
            print('Model saved in {} (quality={:.2f})'.format(save_path, quality))
            data = {'num_updates':num_updates, "model":model, "optim":optim}
            torch.save(data, save_path)
            if self.best_quality < quality:
                print('Model saved as {}'.format(self.best_model_path))
                self.best_quality = quality
                torch.save(data, self.best_model_path)

    def _build_save_path(self, args):
        sub_folder = "{}-{}".format(
            args.algo,
            "rnn" if args.recurrent_policy else "ff"
        )
        save_subdir = os.path.join(args.save_dir, sub_folder)
        return save_subdir

    def save_args(self, file_name='train_args.json', exclude_args=tuple()):
        args = self.args
        folder = self.save_subdir

        save_args = {k:v for k, v in vars(args).items() if k not in exclude_args}
        file_path = os.path.join(folder, file_name)
        ensure_dir(file_path)
        with open(file_path, 'w') as f:
            status = json.dump(save_args, f, sort_keys=True, indent=2)

        print('Train arguments saved in {}'.format(file_path))
        return status


def log_progress(summary,
        curr_update, curr_step, ep_rewards, ep_success, ep_len,
        dist_entropy, value_loss, action_loss,
        fps, loop_fps,
):
    mean_r = np.mean(ep_rewards)
    median_r = np.median(ep_rewards)
    min_r = np.min(ep_rewards)
    max_r = np.max(ep_rewards)
    mean_success = np.mean(ep_success)
    mean_eplen = np.mean(ep_len)
    print(
        "Updates {}, num_steps {}, FPS/Loop FPS {}/{} \n"
        "Last {} episodes:\n  mean/median R {:.2f}/{:.2f}, min/max R {:.1f}/{:.1f}\n"
        "  mean success {:.2f},  mean length {:.1f}\n".format(
            curr_update, curr_step, fps, loop_fps,
            len(ep_rewards), mean_r, median_r,
            min_r, max_r, mean_success, mean_eplen
        )
    )

    if summary is None: return
    summary.add_scalars(
        "Env/reward", {"mean":mean_r, "median":median_r},
        curr_step,
    )
    summary.add_scalar("Env/success", mean_success, curr_step)
    summary.add_scalar("Env/episode_length", mean_eplen, curr_step)

    summary.add_scalar('Loss/enropy', dist_entropy, curr_step)
    summary.add_scalar('Loss/critic', value_loss, curr_step)
    summary.add_scalar('Loss/actor', action_loss, curr_step)

    summary.add_scalars("Performance/FPS", {"total":fps, "loop":loop_fps}, curr_step)


def main():
    args = get_args()
    if args.seed is None:
        args.seed = np.random.randint(1000)
    steps_per_update = args.num_steps*args.num_processes
    args.total_updates = int(args.num_env_steps) // steps_per_update

    print("env_path:", args.env_path)
    print("config_dir:", args.config_dir)
    print("headless:", args.headless)
    print("recurrent_policy:", args.recurrent_policy)
    print()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    #gen_config = ListSampler.create_from_dir(args.config_dir)
    gen_config = SingleConfigGenerator.from_file(
        #"aai_resources/test_configs/MySample2.yaml"
        "aai_resources/default_configs/1-Food.yaml"
    )
    test_gen_config = copy.deepcopy(gen_config)

    envs = make_vec_envs_aai(
        args.env_path, gen_config, args.seed, args.num_processes,
        args.log_dir,  device, allow_early_resets=False, headless=args.headless,
        image_only=len(args.extra_obs) == 0,
    )

    actor_critic = AAIPolicy(
        envs.observation_space,
        envs.action_space,
        base_kwargs={
            'recurrent': args.recurrent_policy,
            'extra_obs': args.extra_obs,
            'hidden_size':512,
            'extra_encoder_dim':128,
            'image_encoder_dim':512
        }
    )

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            acktr=True)

    rollouts = create_storage(
        args.num_steps, args.num_processes,
        envs.observation_space, envs.action_space,
        actor_critic.recurrent_hidden_state_size
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    episode_success = deque(maxlen=100)
    episode_len = deque(maxlen=100)

    start_time = time.time()
    model_saver = DummySaver(args)
    summary = SummaryWriter(os.path.join(model_saver.save_subdir, 'summary'))
    try:
        for curr_update in range(args.total_updates):
            loop_start_time = time.time()
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, curr_update, args.total_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():

                    assert torch.equal(obs['image'], rollouts.obs[step].asdict()['image']), 'woy!! this is strange!'

                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs, #rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)

                for info in infos:
                    if 'episode_reward' in info.keys():
                        episode_rewards.append(info['episode_reward'])
                        episode_success.append(info['episode_success'])
                        episode_len.append(info['episode_len'])

                # If done then clean the history of observations.
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.tensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos]
                )

                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                assert torch.equal(obs['image'], rollouts.obs[-1].asdict()['image']), 'woy!! this is strange!'
                next_value = actor_critic.get_value(
                    obs, #rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()


            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            curr_steps = (curr_update + 1) * args.num_processes * args.num_steps

            if curr_update % args.log_interval == 0 and len(episode_rewards):
                log_progress(
                    summary,curr_update, curr_steps,
                    episode_rewards, episode_success, episode_len,
                    dist_entropy, value_loss, action_loss,
                    fps=int(curr_steps/(time.time()-start_time)),
                    loop_fps=int(steps_per_update/(time.time()-loop_start_time))
                )

                # save for every interval-th episode or for the last epoch
            if len(episode_rewards):
                model_saver.save_model(
                    curr_update, curr_steps,
                    np.mean(episode_rewards), actor_critic
                )

            if (args.eval_interval is not None and len(episode_rewards) > 1
                    and curr_update % args.eval_interval == 0):
                evaluate(actor_critic, args.env_path, test_gen_config, args.seed,
                         args.num_processes, eval_log_dir, device, args.headless)
    finally:
        if summary: summary.close()
        envs.close()


if __name__ == "__main__":
    main()
