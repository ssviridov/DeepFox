
from a2c_ppo_acktr.model import CNNBase, NNBase, Bernoulli,\
    Categorical, DiagGaussian, MLPBase, Flatten, Policy
from torch import nn
import torch as th
from a2c_ppo_acktr.utils import init, conv_output_shape
import numpy as np
import gym

from a2c_ppo_acktr.aai_layers import \
    TemporalAttentionPooling, NaiveHistoryAttention, CachedAttention

class DummyPolicy(Policy):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__() #each we skip Policy's initialization!
        if base_kwargs is None:
            base_kwargs = {}
        assert base is not None, "No default policy!"

        self.base = base(obs_space, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError


class DummyMLP(NNBase):

    def __init__(
            self,
            obs_space,
            policy="ff",
            encoder_size=64,
            hidden_size=None,
    ):
        self._encoder_size = encoder_size
        self._hidden_size = hidden_size if hidden_size else encoder_size
        self._policy_type = policy

        super(DummyMLP, self).__init__(policy=='rnn', self._encoder_size, self._hidden_size)

        self.obs_shapes = self._get_obs_shapes(obs_space)
        self._create_obs_encoder()
        self._create_critic()
        self.train()

    def _get_obs_shapes(self, obs_space):
        if isinstance(obs_space, gym.spaces.Dict):
            return {k: v.shape for k, v in obs_space.spaces.items()}
        else:
            return {"image": obs_space.shape}

    def _create_critic(self):
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self._hidden_size, 1))

    def _create_obs_encoder(self):
        num_inputs = self.obs_shapes['obs'][-1]
        init_ = lambda m: init(
            m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu")
        )

        self.obs_encoder = nn.Sequential(
            init_(nn.Linear(num_inputs, self._encoder_size)), nn.ReLU(),
            #init_(nn.Linear(self._hidden_size, self._hidden_size)), nn.ReLU()
        )

    def forward(self, input, rnn_hxs, masks, **kwargs):
        x = self.obs_encoder(input['obs'])

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPWithAttention(DummyMLP):

    def __init__(
            self,
            obs_space,
            policy="mha",
            encoder_size=64,
            freeze_encoder=False,
    ):

        super(MLPWithAttention, self).__init__(
            obs_space, policy, encoder_size, hidden_size=2*encoder_size,
        )

        assert not self.is_recurrent, "no RRN in my multi-head-attention network!"
        if policy == 'tc':
            self.attention_layer = TemporalAttentionPooling(
                self._encoder_size)
        elif policy == 'mha':
            self.attention_layer = NaiveHistoryAttention(self._encoder_size, 2)
        else:
            raise NotImplementedError("Don't what are you talking about? {}-attention?".format(policy))

        self._freeze_encoder = freeze_encoder
        if self._freeze_encoder:
            print('Encoder is freezed!')
            for p in self.obs_encoder.parameters():
                p.requires_grad=False

    def _flatten_batch(self, input):
        batch_shapes = {}
        flatten_input = {}
        for k,v in input.items():
            n_data_dims =  1 #if k in self._vector_obs else 3
            data_shape = v.shape[-n_data_dims:]
            batch_shapes[k] = v.shape[:-n_data_dims]
            flatten_input[k] = v.view(-1, *data_shape)

        return flatten_input, batch_shapes

    def _unflatten_batch(self, batch_dict, flatten_shapes):
        unflatten = {}
        for k, v in batch_dict.items():
            batch_shape = flatten_shapes[k]
            dim_prod, *data_shape = v.shape
            assert dim_prod == np.prod(flatten_shapes[k]), "Error during flattening a batch!"
            unflatten[k] = v.view(*batch_shape, *data_shape)
        return unflatten

    def forward(self, input, rnn_hxs, masks, **kwargs):
        flatten_input, batch_shapes = self._flatten_batch(input)
        batch_shape = batch_shapes['obs']

        x = self.obs_encoder(flatten_input['obs'])

        x = x.view(*batch_shape, *x.shape[1:])
        x = self.attention_layer(x[:,-1], x[:,:-1])

        return self.critic_linear(x), x, rnn_hxs

class MLPWithCachedAttention(DummyMLP):

    def __init__(
            self,
            obs_space,
            policy="mha",
            encoder_size=64,
            history_len=10,
    ):
        super(MLPWithCachedAttention, self).__init__(
            obs_space, policy, encoder_size, hidden_size=2*encoder_size,
        )
        assert not self.is_recurrent, "no RRN in my multi-head-attention network!"
        if policy == 'tc':
            attention_layer = TemporalAttentionPooling(self._encoder_size)
        elif policy == 'mha':
            attention_layer = NaiveHistoryAttention(self._encoder_size, 2)
        else:
            raise NotImplementedError("Don't what are you talking about? {}-attention?".format(policy))
        self.attention_layer = CachedAttention(attention_layer, history_len)

    def forward(self, input, cached_history, masks, **kwargs):

        x = self.obs_encoder(input['obs'])

        x, cached_history = self.attention_layer(input, cached_history, masks)

        return self.critic_linear(x), x, cached_history
