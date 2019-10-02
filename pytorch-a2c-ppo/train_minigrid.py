import time
from collections import deque

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from dummy_envs.minigrid.minigrid_args import get_args
from dummy_envs.minigrid.minigrid_env import make_vec_minigrid
from dummy_envs.memory_models import DummyPolicy, DummyMLP, MLPWithAttention, MLPWithCachedAttention

from a2c_ppo_acktr.aai_storage import create_storage

from tensorboardX import SummaryWriter
from train_aai import DummySaver, args_to_str

def log_progress(summary,
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
    summary.add_scalar("Env/reward-mean", mean_r, curr_step)
    summary.add_scalar("Env/success", mean_success, curr_step)
    summary.add_scalar("Env/episode-len", mean_eplen, curr_step)

    summary.add_scalar('Loss/enropy', dist_entropy, curr_step)
    summary.add_scalar('Loss/critic', value_loss, curr_step)
    summary.add_scalar('Loss/actor', action_loss, curr_step)

    summary.add_scalar("Performance/FPS", fps, curr_step)


def main():
    args = get_args()
    if args.seed is None:
        args.seed = np.random.randint(1000)

    steps_per_update = args.num_steps*args.num_processes
    args.total_updates = int(args.num_env_steps) // steps_per_update

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    args.base_network_args = {
        'policy':args.policy,
        'encoder_size':args.hidden_size,
        # 'freeze_encoder':False,
    }

    if args.policy in ['rnn', 'ff']:
        BaseNet = DummyMLP

    elif args.policy.startswith('cached'):
        BaseNet = MLPWithCachedAttention
        args.base_network_args['history_len'] = args.frame_stack - 1
        args.frame_stack = 1
    else:
        BaseNet = MLPWithAttention

    envs = make_vec_minigrid(args.game,
        args.num_processes, args.seed, device,
        args.frame_stack,
        image_only_stack=(args.policy not in ['tc', 'mha'])
    )

    actor_critic = DummyPolicy(
        envs.observation_space,
        envs.action_space,
        base=BaseNet, #MLPWithAttention,
        base_kwargs=args.base_network_args
    )
    args.network_architecture = repr(actor_critic)

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
        actor_critic.internal_state_shape
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=150)
    episode_success = deque(maxlen=150)
    episode_len = deque(maxlen=150)

    print(args_to_str(args))

    start_time = time.time()
    model_saver = DummySaver(args)
    summary = SummaryWriter(args.summary_dir)

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
                    value, action, action_log_prob, internal_state = actor_critic.act(
                        obs, #rollouts.obs[step], rollouts.obs[step-obs_stack+1:step+1],
                        rollouts.internal_states[step],
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

                rollouts.insert(obs, internal_state, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
    #            assert torch.equal(obs['image'], rollouts.obs[-1].asdict()['image']), 'woy!! this is strange!'
                next_value = actor_critic.get_value(
                    obs, #rollouts.obs[-1],
                    rollouts.internal_states[-1],
                    rollouts.masks[-1]).detach()


            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            curr_steps = (curr_update + 1) * args.num_processes * args.num_steps

            if curr_update % args.log_interval == 0 and len(episode_rewards):
                log_progress(
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
