from collections import OrderedDict

from a2c_ppo_acktr.model import CNNBase, NNBase, Bernoulli,\
    Categorical, DiagGaussian, MLPBase, Flatten, Policy
from .utils import mlp_body
import gym
from torch import nn
import torch as th
from a2c_ppo_acktr.utils import init, default_init, conv_output_shape
import numpy as np
from .aai_layers import NaiveHistoryAttention, TemporalAttentionPooling, CachedAttention
import torch.nn.functional as F
import logging


class AAIPolicy(Policy):

    def __init__(
            self, obs_space,
            action_space,
            base=None,
            base_kwargs=None,
    ):
        super(Policy, self).__init__() #each we skip Policy's initialization!
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = ImageVecMap

        self.base = base(
            obs_space,
            **base_kwargs
        )

        # Kostrikov decided to merge state-encoder and critic-head inside
        # one module(base) which he then wrapps with this module that defines
        # actor-head separately. This is a questionable design choice imo.
        # But I'm too lazy to change this
        self.dist = Categorical(
            self.base.output_size,
            action_space.n,
            **self.base.head_kwargs #comment above
        )


class AAIBody(nn.Module):
    def __init__(self, body_type, input_size, **kwargs):

        super(AAIBody, self).__init__()
        self._body_type = body_type

        # these will be initialized in the _init_{something} functions:
        self._sequential = None  # bool
        self._hidden_size = None  # int
        self._internal_state_shape = None  # tuple of ints
        self._body_forward = None  # a method that processes input from encoders and returns a final
        # embedding for policy and value heads

        if self._body_type == 'rnn':
            self._init_rnn(input_size, kwargs)

        elif self._body_type.startswith("cached"):
            self._init_attn(input_size, kwargs)

        elif self._body_type == 'ff':
            #ok here is a question:
            #if policy is feedforward should we add one more feedforward layer?
            # or just ignore it an attach output from various encoders directly to policy and value heads?
            #(like Kostrikov did in his code)
            self._init_none(input_size, kwargs)

        else:
            raise NotImplementedError("recurrency type {} is unknown!".format(self._body_type))

        self._check_init(kwargs)

    def _init_rnn(self, input_size, kwargs):

        self._sequential = True
        self._hidden_size = kwargs.pop('hidden_size', input_size)

        self.gru = nn.GRU(input_size, self._hidden_size)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self._internal_state_shape = (self._hidden_size,)
        self._body_forward = self._forward_gru

    def _init_attn(self, input_size, kwargs):

        self._sequential = True
        self._hidden_size = 2 * kwargs.pop('hidden_size', input_size)

        if self._body_type.endswith('tc'):
            attn = TemporalAttentionPooling(input_size)

        elif self._body_type.endswith('mha'):
            attn_heads = kwargs.pop('attention_heads', 3)
            attn = NaiveHistoryAttention(input_size, attn_heads)

        else:
            raise NotImplementedError("{}? What is that?".format(self._body_type))

        self._memory_len = kwargs.pop('memory_len', 10)
        self.attention_layer = CachedAttention(attn, self._memory_len)

        self._internal_state_shape = (self._memory_len, self._hidden_size)
        self._body_forward = self._forward_attn

    def _init_none(self, input_size, kwargs):
        self._sequential = False
        self._internal_state_shape = (1,)
        self._hidden_size = input_size
        self._body_forward = self._forward_identity

    def _check_init(self, kwargs):
        # check that all fields are initialized!
        for field in ('_sequential', '_hidden_size', '_internal_state_shape', '_body_forward'):
            assert getattr(self, field) is not None, "self.{} is not initialized!".format(field)

            # check for unused parameters:
        if kwargs:
            logging.warning(
                "AAIBody: Following arguments were not used during initialization:\n {}".format(kwargs)
            )

    @property
    def is_sequential(self):
        return self._sequential

    @property
    def internal_state_shape(self):
        return self._internal_state_shape

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = th.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

    def _forward_attn(self, x, cached_history, masks):
        N = cached_history.size(0)
        obs_batch_size = x.size(0)
        if obs_batch_size == N:
            x, cached_history = self.attention_layer(x, cached_history, masks)
            return x, cached_history
        else:
            T = obs_batch_size // N
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)
            outputs = []
            for t in range(T):
                out, cached_history = self.attention_layer(x[t], cached_history, masks[t])
                outputs.append(out)

            outputs = th.cat(outputs, dim=0)
            return outputs, cached_history

    def _forward_identity(self, x, hxs, masks):
        return x, hxs,


class ImageVecMap(AAIBody):
    """
    New shiny IVM arch, now you cant build multi-layered critic and policy heads and use attention
    """
    def __init__(
            self,
            obs_space,
            body_type='ff',
            extra_obs=None,
            body_kwargs=None,
            head_kwargs=None,
            map_dim=384,
            image_dim=512,
    ):
        assert 'visited' in extra_obs, "It is better to choose different network if you are not planning to use visited map"
        self._image_encoder_dim = image_dim
        self._map_encoder_dim = map_dim if 'visited' in extra_obs else 0
        # if there is no extra_obs then we don't need an extra_encoder!
        self._extra_obs = extra_obs
        self._obs_shapes = self._get_obs_shapes(obs_space)
        self._vector_obs = [k for k in self._extra_obs if len(self._obs_shapes[k]) <= 2] #images has 3+ dims
        self._vector_dim =  sum(self._obs_shapes[k][0] for k in self._vector_obs)

        self._total_encoder_dim = self._image_encoder_dim + self._map_encoder_dim + self._vector_dim

        self.head_kwargs = head_kwargs if head_kwargs else {}

        super(ImageVecMap, self).__init__(body_type, self._total_encoder_dim, **body_kwargs)

        self._build_network()
        self.train()

    def _build_network(self):
        self._create_image_encoder()
        self._create_map_encoder()
        self._create_critic()

    def _get_obs_shapes(self, obs_space):
        if isinstance(obs_space, gym.spaces.Dict):
            return {k: v.shape for k, v in obs_space.spaces.items()}
        else:
            return {"image": obs_space.shape}

    def _create_critic(self):
        hs = self.head_kwargs.get('hidden_sizes', tuple())
        nl = self.head_kwargs.get('nl', 'relu')

        if hs:
            critic_head = mlp_body(self._hidden_size, hs, nl)
            final_layer = default_init(nn.Linear(hs[-1], 1))
            critic_head.add_module(str(len(critic_head)), final_layer)
        else:
            critic_head = default_init(nn.Linear(self._hidden_size, 1))

        self.critic_head = critic_head

    def _create_image_encoder(self):
        num_channels = self._obs_shapes['image'][0]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, self._image_encoder_dim)), nn.ReLU()
        )
        #self.last_fc = init_(nn.Linear(self._total_encoder_dim, self._hidden_size))

    def _create_map_encoder(self):
        if 'visited' in self._extra_obs:
            num_channels, H, W = self._obs_shapes['visited']
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), nn.init.calculate_gain('relu'))
            if H > 16:
                c_kernel, c_stride = 4, 2
                c_pad = (H % 2)
                c_channels = 64
            else:
                c_kernel, c_stride, c_pad = 3, 1, 0
                c_channels = 16

            encoder = OrderedDict([
                ("conv1", init_(nn.Conv2d(num_channels, 32, c_kernel, c_stride,padding=c_pad))),
                ("relu1", nn.ReLU()),  # was num_channes -> 16
                ('conv2', init_(nn.Conv2d(32, 32, c_kernel, c_stride, padding=c_pad))),
                ('relu2', nn.ReLU()),  # was 16, -> 32
                ('conv3', init_(nn.Conv2d(32, c_channels, 3, 1))),
                #('maxpool3'), nn.MaxPool2d(2,2),
                ('relu3', nn.ReLU()),  # was AvgPool2D without relu
                ('faltten4', Flatten()),
            ])
            conv_features = conv_output_shape(self._obs_shapes['visited'], encoder)[0]
            encoder['fc5'] = init_(nn.Linear(conv_features, self._map_encoder_dim))
            encoder['relu5'] = nn.ReLU()
            self.map_encoder = nn.Sequential(encoder)

    def concat_vector_obs(self, obs):
        return th.cat([obs[k] for k in self._vector_obs], dim=1)

    def forward(self, input, memory, masks, **kwargs):
        x_img = self.image_encoder(input['image'] / 255.0)
        map_embed = self.map_encoder(input['visited'])
        vector_obs = self.concat_vector_obs(input)
        x = th.cat([x_img, map_embed, vector_obs], dim=1)

        x, memory = self._body_forward(x, memory, masks)

        return self.critic_linear(x), x, memory


"""
class NaiveAttentionIVM(ImageVecMap):
    #This thing doesn't work for now
    def __init__(
            self,
            obs_space,
            extra_obs=None,
            policy="mha",
            hidden_size=None,
            extra_encoder_dim=512,
            image_encoder_dim=512,
    ):
        super(NaiveAttentionIVM, self).__init__(
            obs_space,
            extra_obs,
            policy,
            (extra_encoder_dim+image_encoder_dim)*2,
            extra_encoder_dim,
            image_encoder_dim,
        )
        if policy == 'tc':
            self.attention_layer = TemporalAttentionPooling(self._total_encoder_dim)
        elif policy == 'mha':
            self.attention_layer = NaiveHistoryAttention(self._total_encoder_dim, 2)
        else:
            raise NotImplementedError("Don't what are you talking about? {}-attention?".format(policy))

        #self.attention_layer = TemporalAttentionPooling(self._total_encoder_dim)
        #NaiveHistoryAttention(self._total_encoder_dim, 4)

    def _flatten_batch(self, input):
        batch_shapes = {}
        flatten_input = {}
        for k,v in input.items():
            n_data_dims =  1 if k in self._vector_obs else 3
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
"""
