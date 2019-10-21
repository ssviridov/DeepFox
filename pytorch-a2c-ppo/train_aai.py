import os
import time
import datetime
from collections import deque

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.aai_arguments import get_args
from a2c_ppo_acktr.aai_wrapper import make_vec_envs_aai
from a2c_ppo_acktr.aai_models import AAIPolicy, ImageVecMap

from a2c_ppo_acktr.aai_storage import create_storage

import a2c_ppo_acktr.aai_config_generator as configs
import json
from tensorboardX import SummaryWriter


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
        if not args.restart:
            self.save_args()
        self.model_path = os.path.join(self.save_subdir, "under-{}0M-steps.pt")
        self.best_model_path = os.path.join(self.save_subdir, "best.pt")

    def save_model(self, num_updates, total_step, quality, model, optim=None):
        if num_updates % self.save_every_updates == 0 or num_updates == (self.args.total_updates-1):
            # 9950k goes into under-10M-steps.pt
            # 10200k goes into under-20M-steps.pt
            lt_steps = str((total_step // 10000000) + 1)
            save_path = self.model_path.format(lt_steps)
            print('Model saved in {} (quality={:.2f})'.format(save_path, quality))
            data = {'num_updates':num_updates, "total_step":total_step, "model": model, "optim": optim.state_dict()}
            torch.save(data, save_path)
            if self.best_quality < quality:
                print('Model saved as {}'.format(self.best_model_path))
                self.best_quality = quality
                torch.save(data, self.best_model_path)

    def _build_save_path(self, args):
        assert hasattr(args, 'experiment_tag'), 'Please specify --experiment-tag, -et'
        save_subdir = os.path.join(args.save_dir, args.experiment_tag)
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
        curr_update, curr_step, ep_rewards, ep_scaled_rewards, ep_success, ep_len,
        ep_visited, ep_oracle_r, dist_entropy, value_loss, action_loss,
        fps, loop_fps,
):
    mean_r = np.mean(ep_rewards)
    if ep_scaled_rewards:
        mean_scaled_r = np.mean(ep_scaled_rewards)
    else:
        mean_scaled_r = 0.
    median_r = np.median(ep_rewards)
    min_r = np.min(ep_rewards)
    max_r = np.max(ep_rewards)
    mean_success = np.mean(ep_success)
    mean_eplen = np.mean(ep_len)
    mean_visited = np.mean(ep_visited)
    mean_oracle_r = np.mean(ep_oracle_r)
    print(
        "Updates {}, num_steps {}, FPS/Loop FPS {}/{} \n"
        "Last {} episodes:\n  mean/median R {:.2f}/{:.2f}, E[scaled-R] {:.2f}, min/max R {:.1f}/{:.1f}\n"
        "  E[success] {:.2f},  E[length] {:.1f}, E[visted] {:.1f}, E[oracle-R] {:.2f}\n".format(
            curr_update, curr_step, fps, loop_fps,
            len(ep_rewards), mean_r, median_r, mean_scaled_r,
            min_r, max_r, mean_success, mean_eplen, mean_visited, mean_oracle_r
        )
    )

    if summary is None: return
    summary.add_scalar("Env/r-mean", mean_r, curr_step)
    if ep_scaled_rewards:
        summary.add_scalar("Env/r-scaled-mean", mean_scaled_r, curr_step)

    summary.add_scalar("Env/r-median", median_r, curr_step)
    summary.add_scalar("Env/success", mean_success, curr_step)
    summary.add_scalar("Env/episode-len", mean_eplen, curr_step)

    summary.add_scalar('GridOracle/visited-cells', mean_visited, curr_step)
    summary.add_scalar('GridOracle/mean-oracle-r', mean_oracle_r, curr_step)

    summary.add_scalar('Loss/entropy', dist_entropy, curr_step)
    summary.add_scalar('Loss/critic', value_loss, curr_step)
    summary.add_scalar('Loss/actor', action_loss, curr_step)

    summary.add_scalar("Performance/total-fps", fps, curr_step)
    summary.add_scalar("Performance/loop-fps", loop_fps, curr_step)


def args_to_str(args):
    lines = ['','ARGUMENTS:']
    newline = os.linesep
    args = vars(args)
    for key in sorted(args.keys()):
        lines.append('    "{0}": {1}'.format(key, args[key]))
    return newline.join(lines)


def make_model_kwargs(args):
    body_kwargs = dict()

    if args.policy == 'rnn':
        body_kwargs['hidden_size']=512

    elif args.policy.startswith('cached'):
        body_kwargs['memory_len']= getattr(args, 'memory_len', 20)
        body_kwargs['attention_heads'] = getattr(args, 'attention_heads', 9) #ignored with cached_tc
        # body_kwargs['residual']=True
        # body_kwargs['layer_norm']=True

    model_kwargs = dict(
        body_type=args.policy,
        extra_obs=args.extra_obs,
        head_kwargs=dict(hidden_sizes=(512,)), #nl='tanh'),  # defaults: (hidden_sizes=tuple(), nl='relu')
        body_kwargs=body_kwargs,
        map_dim=384,  # 'extra_encoder_dim':384,
        image_dim=384,
    )
    return model_kwargs

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
    device_str = "cuda:{}".format(args.device) if args.cuda else "cpu"
    device = torch.device(device_str)

    #Load config files and determine a method of config sampling:
    gen_config = configs.HierarchicalSampler.create_from_dir(args.config_dir)

    gen_config = configs.RandomizedGenerator.create(
        gen_config, args.rnd_blackout, args.rnd_object, args.rnd_color
    )
    #gen_config = SingleConfigGenerator.from_file(
    #    "aai_resources/new_configs/mazes/chess_walls.yaml")

    args.real_oracle_args = dict(
        oracle_type="angles",
        oracle_reward=args.oracle_reward,
        num_angles=args.oracle_num_angles,
        cell_side=args.oracle_cell_side,
        # trace_decay=0.992, # randomly
    )
    args.classifier_args = None
    if 'objects' in args.extra_obs:
        args.classifier_args = dict(
            model_path=args.clf_path,
            threshold=args.clf_threshold,
            device='cpu' if args.clf_cpu else device_str
        )

    envs = make_vec_envs_aai(
        args.env_path, gen_config, args.seed, args.num_processes,
        device, num_frame_stack=args.frame_stack,
        headless=args.headless,
        grid_oracle_kwargs=args.real_oracle_args,
        classifier_kwargs=args.classifier_args,
        image_only=len(args.extra_obs) == 0,
        docker_training=args.docker_training,
        reduced_actions=args.reduced_actions,
        scale_reward=args.scale_reward
    )

    #Create Agent:
    if not args.restart:
        args.base_network_args = make_model_kwargs(args)
        network_class = ImageVecMap
        args.network_cls = network_class.__name__

        actor_critic = AAIPolicy(
            envs.observation_space,
            envs.action_space,
            base=network_class,
            base_kwargs=args.base_network_args
        )

    else:
        print("Loading model...")
        data = torch.load(args.restart)
        actor_critic = data['model']
        updates_done = data['num_updates']
        args.total_updates = int((args.total_updates - updates_done)*args.num_steps*args.num_processes) // steps_per_update

    actor_critic.to(device)
    args.network_architecture = repr(actor_critic)

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

    if args.restart and (data['optim'] is not None):
        agent.optimizer.load_state_dict(data['optim'])

    rollouts = create_storage(
        args.num_steps, args.num_processes,
        envs.observation_space, envs.action_space,
        actor_critic.internal_state_shape
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=200)
    if args.scale_reward:
        episode_scaled_rewards = deque(maxlen=200)
    else:
        episode_scaled_rewards = None
    episode_success = deque(maxlen=200)
    episode_len = deque(maxlen=200)
    episode_visited = deque(maxlen=200)
    episode_oracle_rewards = deque(maxlen=200)

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
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs, #rollouts.obs[step],
                        rollouts.internal_states[step],
                        rollouts.masks[step])

                #/no_grad
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                for info in infos:
                    if 'episode_reward' in info.keys():
                        #print('TRAIN: EPISODE IS DONE!')
                        episode_rewards.append(info['episode_reward'])
                        episode_success.append(info['episode_success'])
                        episode_len.append(info['episode_len'])
                        if "grid_oracle" in info:
                            episode_visited.append(info["grid_oracle"]["n_visited"])
                            episode_oracle_rewards.append(info["grid_oracle"]["episode_r"])

                    if args.scale_reward:
                        episode_scaled_rewards.append(info["episode_scaled_reward"])

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
                    rollouts.internal_states[-1],
                    rollouts.masks[-1]).detach()


            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            if args.restart:
                curr_update_old = curr_update
                curr_update = curr_update + updates_done

            curr_steps = (curr_update + 1) * args.num_processes * args.num_steps

            if curr_update % args.log_interval == 0 and len(episode_rewards):
                if args.restart:
                    fps = int(((curr_update_old + 1) * args.num_processes * args.num_steps)/(time.time()-start_time))
                else:
                    fps = int(curr_steps/(time.time()-start_time))
                log_progress(
                    summary, curr_update, curr_steps,
                    episode_rewards, episode_scaled_rewards, episode_success, episode_len,
                    episode_visited, episode_oracle_rewards,
                    dist_entropy, value_loss, action_loss,
                    fps=fps,
                    loop_fps=int(steps_per_update/(time.time()-loop_start_time))
                )

                #if curr_steps > 5000000:
                #    break

            # save for every interval-th episode or for the last epoch
            if len(episode_rewards) == episode_rewards.maxlen:
                model_saver.save_model(
                    curr_update, curr_steps,
                    np.mean(episode_success), actor_critic, agent.optimizer
                )

    finally:
        if summary: summary.close()
        envs.close()


if __name__ == "__main__":
    main()
