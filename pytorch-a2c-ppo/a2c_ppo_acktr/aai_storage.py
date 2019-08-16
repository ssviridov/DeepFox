import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from .storage import _flatten_helper, RolloutStorage
from typing import List
import gym

def create_storage(
        num_steps,
        num_processes,
        obs_space,
        action_space,
        recurrent_hidden_state_size
):
    if isinstance(obs_space, gym.spaces.Dict):
        storage_cls = RolloutStorageWithDictObs
    else:
        storage_cls = RolloutStorage

    rollouts = storage_cls(
        num_steps, num_processes,
        obs_space, action_space,
        recurrent_hidden_state_size
    )
    return rollouts

class DictAsTensor(object):

    @staticmethod
    def from_dict(dict_of_tenors):
        return DictAsTensor(dict_of_tenors)

    @staticmethod
    def create_empty(dict_of_shapes, repeat_shape=()):
        tensor = {}
        for k, shape in dict_of_shapes.items():
            tensor[k] = torch.zeros(*repeat_shape, *shape)
        return DictAsTensor(tensor)

    @staticmethod
    def stack(list_of_dict_as_tensors, dim):
        keys = list(list_of_dict_as_tensors[0].tensor.keys())
        stacked = {}
        for k in keys:
            stacked[k] = torch.stack(
                [dat.tensor[k] for dat in list_of_dict_as_tensors], dim=dim
            )
        return DictAsTensor.from_dict(stacked)

    def __init__(self, dict_of_tensors):
        """Makes a shallow copy of the dictionary of tensors"""
        super(DictAsTensor, self).__init__()
        self.tensor = {}
        for k, t in dict_of_tensors.items():
            self.tensor[k] = t

    def to(self,  *args, **kwargs):
        result = {k:t.to(*args, **kwargs) for k, t in self.tensor.items()}
        return self.from_dict(result)

    def __getitem__(self, key):
        """
        Returns a new DictOfTensors with results of indexing or slicing.
        Warning: it may store just a views of the original tensors, or a full copies depending on the slicing you are using!
        """
        return self.from_dict({k:t[key] for k, t in self.tensor.items()})

    def copy_(self, dict_of_tensors):
        # if this is not a real dict then it is a DictAsTensor
        if not hasattr(dict_of_tensors, "items"):
            dict_of_tensors = dict_of_tensors.tensor

        for k,t in self.tensor.items():
            t.copy_(dict_of_tensors[k])

    def flatten_view(self, n_first_dims):
        tensor = {}
        for k, t in self.tensor.items():
            tensor[k] = t.view(-1, *t.shape[n_first_dims:])

        return self.from_dict(tensor)

    def size(self):
        return {k:v.size() for k, v in self.tensor.items()}

    def asdict(self):
        return dict(self.tensor)


class RolloutStorageWithDictObs(object):

    def __init__(self, num_steps, num_processes, obs_space, action_space,
                 recurrent_hidden_state_size):
        # Be really cautions with DictAsTensor.
        # It's only simulate behavior of the real tensors on the operations
        # that i use here
        obs_shapes = {k:sp.shape for k,sp in obs_space.spaces.items()}
        self.obs = DictAsTensor.create_empty(
            obs_shapes, repeat_shape=(num_steps+1, num_processes)
        )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].flatten_view(n_first_dims=2)
            obs_batch = obs_batch[indices]
            obs_batch = obs_batch.asdict()
            #obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = DictAsTensor.stack(obs_batch, 1)

            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = obs_batch.flatten_view(n_first_dims=2)
            obs_batch = obs_batch.asdict()
            # obs_batch = _flatten_helper(T, N, obs_batch)

            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

# rollouts.obs[0].copy_(obs)
# rollouts.to(device)

# rollouts.obs[step],
# rollouts.recurrent_hidden_states[step],
# rollouts.masks[step]

# rollouts.insert

# rollouts.obs[-1],
# rollouts.recurrent_hidden_states[-1],
# rollouts.masks[-1]

#rollouts.compute_returns(
#   next_value, args.use_gae, args.gamma,
#    args.gae_lambda,
#    args.use_proper_time_limits
#)

# ==== PPO UPDATE ========:
# rollouts.returns[:-1] - rollouts.value_preds[:-1]
# rollouts.recurrent_generator(
# rollouts.feed_forward_generator(
#=========================

# ==== A2C UPDATE ========:
#  rollouts.obs.size()[2:]
#  rollouts.obs[:-1].view(-1, *obs_shape)
#=========================

#rollouts.after_update()
