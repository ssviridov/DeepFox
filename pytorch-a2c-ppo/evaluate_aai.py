import numpy as np
import torch

from a2c_ppo_acktr.aai_wrapper import make_vec_envs_aai


def evaluate(actor_critic, env_path, gen_config, seed, num_processes,
             device, headless):

    envs = make_vec_envs_aai(
        env_path, gen_config, seed+num_processes+10, num_processes,
        device, headless=headless
    )
    #eval_envs = make_vec_envs_aai(
    #   env_name, seed + num_processes, num_processes,
    #    None, eval_log_dir, device, True
    #)
    eval_episode_rewards = []

    obs = envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.internal_state_shape, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True
            )

        # Obser reward and next obs
        obs, _, done, infos = envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
