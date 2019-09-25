import os
import time
from collections import deque

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from dummy_envs.memory_env_args import get_args
from dummy_envs.memory_env import make_vec_dummy_memory
from dummy_envs.memory_models import DummyPolicy, DummyMLP, MLPWithAttention, MLPWithCachedAttention

from a2c_ppo_acktr.aai_storage import create_storage

from tensorboardX import SummaryWriter
from train_aai import DummySaver, ensure_dir, args_to_str

def log_progress(experiment_tag, summary,
        curr_update, curr_step, ep_rewards, ep_success, ep_len,
        dist_entropy, value_loss, action_loss, fps, loop_fps,
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
            min_r, max_r, mean_success, mean_eplen,
        )
    )

    if summary is None: return
    summary.add_scalars("Env/mean_reward", {experiment_tag:mean_r}, curr_step)

    summary.add_scalars("Env/success", {experiment_tag:mean_success}, curr_step)
    summary.add_scalars("Env/episode_length", {experiment_tag:mean_eplen}, curr_step)

    summary.add_scalars('Loss/enropy', {experiment_tag:dist_entropy}, curr_step)
    summary.add_scalars('Loss/critic', {experiment_tag:value_loss}, curr_step)
    summary.add_scalars('Loss/actor', {experiment_tag:action_loss}, curr_step)

    summary.add_scalars("Performance/FPS", {experiment_tag:fps}, curr_step)

def summary_path(save_dir):
    experiment_dir = os.path.dirname(os.path.relpath(save_dir, "pretrained"))
    return os.path.join('pretrained', experiment_dir, 'summaries')

def main():
    args = get_args()
    assert args.num_steps <= args.episode_length+1, "We don't need this until we get to meta-rl!"
    if args.seed is None:
        args.seed = np.random.randint(1000)
    steps_per_update = args.num_steps*args.num_processes
    args.total_updates = int(args.num_env_steps) // steps_per_update
    experiment_tag = os.path.basename(args.save_dir)


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_dummy_memory(
        args.num_processes, device, args.seed,
        args.episode_length, args.frame_stack if args.frame_stack > 1 else None,
    )

    actor_critic = DummyPolicy(
        envs.observation_space,
        envs.action_space,
        base=DummyMLP if args.policy in ['rnn', 'ff'] else MLPWithCachedAttention,
        base_kwargs={
            'policy': args.policy,
            'encoder_size':args.hidden_size,
        }
    )
    print("MODEL:")
    print(actor_critic)
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

    print(args_to_str(args))

    start_time = time.time()
    model_saver = DummySaver(args)

    summary = SummaryWriter(summary_path(args.save_dir))
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

                    #assert torch.equal(obs['image'], rollouts.obs[step].asdict()['image']), 'woy!! this is strange!
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs, #rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                #/no_grad
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
    #            assert torch.equal(obs['image'], rollouts.obs[-1].asdict()['image']), 'woy!! this is strange!'
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
                    experiment_tag,
                    summary,curr_update, curr_steps,
                    episode_rewards, episode_success,
                    episode_len, dist_entropy, value_loss, action_loss,
                    fps=int(curr_steps/(time.time()-start_time)),
                    loop_fps=int(steps_per_update/(time.time()-loop_start_time))
                )

                # save for every interval-th episode or for the last epoch
            if len(episode_rewards) == episode_rewards.maxlen:
                model_saver.save_model(
                    curr_update, curr_steps,
                    np.mean(episode_success), actor_critic
                )

    finally:
        if summary: summary.close()
        envs.close()


if __name__ == "__main__":
    main()
