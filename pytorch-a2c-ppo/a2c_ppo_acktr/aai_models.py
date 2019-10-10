from collections import OrderedDict

from a2c_ppo_acktr.model import CNNBase, NNBase, Bernoulli,\
    Categorical, DiagGaussian, MLPBase, Flatten, Policy
from .utils import mlp_body
import gym
from torch import nn
import torch as th
from a2c_ppo_acktr.utils import init, default_init, conv_output_shape
import numpy as np
from .aai_layers import NaiveHistoryAttention, TemporalAttentionPooling
import torch.nn.functional as F

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
            base = ImageVecMap3

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


class ImageVecMapBase(NNBase):

    def __init__(
            self,
            obs_space,
            extra_obs=None,
            policy="ff",
            hidden_size=512,
            extra_encoder_dim=512,
            image_encoder_dim=512,
            head_kwargs=None,
    ):

        self._image_encoder_dim = image_encoder_dim
        # if there is no extra_obs then we don't need an extra_encoder!
        self._extra_encoder_dim = extra_encoder_dim if extra_obs else 0
        self._total_encoder_dim = self._image_encoder_dim + self._extra_encoder_dim
        # if recurrent is False there will be no layer after encoders ouputs:
        self._hidden_size = hidden_size if policy!='ff' else self._total_encoder_dim
        self._extra_obs = extra_obs
        self._policy_type = policy
        self.head_kwargs = head_kwargs if head_kwargs else {}

        super(ImageVecMapBase, self).__init__(
            policy=='rnn',
            self._total_encoder_dim,
            self._hidden_size
        )

        self.obs_shapes = self._get_obs_shapes(obs_space)
        self._image_only_obs = isinstance(obs_space, gym.spaces.Box)

        self._create_image_encoder()

        if self._extra_obs:
            self._create_extra_encoder()

        self._create_critic()

        self.train()

    def _get_obs_shapes(self, obs_space):
        if isinstance(obs_space, gym.spaces.Dict):
            return {k: v.shape for k, v in obs_space.spaces.items()}
        else:
            return {"image": obs_space.shape}

    def _create_critic(self):
        hs = self.head_kwargs.get('hidden_sizes',tuple())
        nl = self.head_kwargs.get('nl', 'relu')
        raise NotImplementedError()

        self.critic_linear = default_init(nn.Linear(self._hidden_size, 1))

    def _create_image_encoder(self):
        num_channels = self.obs_shapes['image'][-3]
        init_ = lambda m: default_init(m, 'relu')

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, self._image_encoder_dim)), nn.ReLU()
        )

    def _create_map_encoder(self):
        num_channels, H, W = self.obs_shapes['visited'][-3:]
        init_ = lambda m: default_init(m, 'relu')

        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(num_channels, 16, 4, 2)), nn.ReLU(),
            init_(nn.Conv2d(16, 32, 3, 1)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            Flatten()
        )

    def _create_extra_encoder(self):
        out_features = self._extra_encoder_dim
        in_features = 0
        if 'visited' in self._extra_obs:
            self._create_map_encoder()
            in_features += conv_output_shape(
                self.obs_shapes['visited'][-3:],
                self.map_encoder
            )[0]
        else:
            self.map_encoder = None
        # all images has at least 3 dims,
        # vectors would have 2(if stacked along time) or 1
        self._vector_obs = [k for k in self._extra_obs if len(self.obs_shapes[k]) < 3]
        for k in self._vector_obs:
            in_features += self.obs_shapes[k][-1]


        self.extra_linear = nn.Sequential(
            default_init(
                nn.Linear(in_features, out_features),
                'relu'
            ),
            nn.ReLU()
        )

    def extra_encoder(self, obs):
        input = th.cat([obs[k] for k in self._vector_obs], dim=1)
        if self.map_encoder:
            map_embed = self.map_encoder(obs['visited'])
            input = th.cat([input, map_embed], dim=1)
        return self.extra_linear(input)

    def forward(self, input, rnn_hxs, masks, **kwargs):

        if self._image_only_obs:
            x = self.image_encoder(input / 255.0)
        else:
            x_img = self.image_encoder(input['image'] / 255.0)
            x_extra = self.extra_encoder(input)
            x = th.cat([x_img, x_extra], dim=1)

        if self.is_sequential:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class ImageVecMap2(NNBase):
    """
    Difference with ImageVecMapBase is that ImageVecMap2 receives all vector observations before the last layer
    """
    def __init__(
            self,
            obs_space,
            extra_obs=None,
            recurrent=False,
            hidden_size=512,
            map_dim=512,
            image_dim=512,
            dropout=0.0
    ):
        assert 'visited' in extra_obs, "It is better to choose different network if you are not planning to use visited map"
        self.dropout=0.0
        self._image_encoder_dim = image_dim
        self._map_encoder_dim = map_dim
        # if there is no extra_obs then we don't need an extra_encoder!
        self._extra_obs = extra_obs
        self._obs_shapes = self._get_obs_shapes(obs_space)
        self._vector_obs = [k for k in self._extra_obs if len(self._obs_shapes[k]) == 1]
        self._vector_dim =  sum(self._obs_shapes[k][0] for k in self._vector_obs)

        self._total_encoder_dim = self._image_encoder_dim + self._map_encoder_dim + self._vector_dim
        # if recurrent is False there will be no layer after encoders ouputs:
        self._hidden_size = hidden_size if recurrent else self._total_encoder_dim


        super(ImageVecMap2, self).__init__(
            recurrent,
            self._total_encoder_dim,
            self._hidden_size
        )

        self._create_image_encoder()
        self._create_map_encoder()
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

    def forward(self, input, rnn_hxs, masks, **kwargs):
        x_img = self.image_encoder(input['image'] / 255.0)
        map_embed = self.map_encoder(input['visited'])
        vector_obs = self.concat_vector_obs(input)
        x = th.cat([x_img, map_embed, vector_obs], dim=1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


def create_fc_layer(input_dim, output_dim, nl='relu'):
    init_ = lambda m:init(m, nn.init.orthogonal_, lambda x:nn.init.
                          constant_(x, 0), nn.init.calculate_gain(nl))
    return init_(nn.Linear(input_dim, output_dim))


class ImageVecMap3(ImageVecMap2):
    """
    Have 2-layered heads instead of 1-layered heads in ImageVecMapBase/2
    """
    def __init__(
            self,
            obs_space,
            extra_obs=None,
            recurrent=False,
            hidden_size=512,
            map_dim=512,
            image_dim=512,
            dropout=0.0
    ):
        assert 'visited' in extra_obs, "It is better to choose different network if you are not planning to use visited map"
        self.dropout=0.0
        self._image_encoder_dim = image_dim
        self._map_encoder_dim = map_dim
        # if there is no extra_obs then we don't need an extra_encoder!
        self._extra_obs = extra_obs
        self._obs_shapes = self._get_obs_shapes(obs_space)
        self._vector_obs = [k for k in self._extra_obs if len(self._obs_shapes[k]) == 1]
        self._vector_dim =  sum(self._obs_shapes[k][0] for k in self._vector_obs)

        self._total_encoder_dim = self._image_encoder_dim + self._map_encoder_dim + self._vector_dim
        # if recurrent is False there will be no layer after encoders ouputs:
        self._hidden_size = hidden_size #if recurrent else self._total_encoder_dim


        super(ImageVecMap2, self).__init__(
            recurrent,
            self._total_encoder_dim,
            self._hidden_size
        )

        self._create_image_encoder()
        self._create_map_encoder()
        self.actor_linear = create_fc_layer(self._total_encoder_dim, self._hidden_size)
        self._create_critic()
        self.train()

    def _create_critic(self):
        self.crtic_linear1 = create_fc_layer(self._total_encoder_dim, self._hidden_size)
        init_ = lambda m:init(m, nn.init.orthogonal_, lambda x:nn.init.constant_(x, 0))
        self.critic_linear2 = init_(nn.Linear(self._hidden_size, 1))

    def forward(self, input, rnn_hxs, masks, **kwargs):
        x_img = self.image_encoder(input['image'] / 255.0)
        map_embed = self.map_encoder(input['visited'])
        vector_obs = self.concat_vector_obs(input)
        x = th.cat([x_img, map_embed, vector_obs], dim=1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        v_pred = self.critic_linear2(F.relu(self.crtic_linear1(x)))
        actor_scores = F.relu(self.actor_linear(x))
        return v_pred, actor_scores, rnn_hxs



class AttentionIVM(ImageVecMapBase):

    def __init__(
            self,
            obs_space,
            extra_obs=None,
            policy="mha",
            hidden_size=None,
            extra_encoder_dim=512,
            image_encoder_dim=512,
    ):
        assert hidden_size is None, "hidden_size=(extra_encoder_dim+image_encoder_dim)*2"

        super(AttentionIVM, self).__init__(
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

    def forward(self, input, rnn_hxs, masks, **kwargs):
        flatten_input, batch_shapes = self._flatten_batch(input)
        batch_shape = batch_shapes['image']

        if self._image_only_obs:
            x = self.image_encoder(flatten_input / 255.0)
        else:
            x_img = self.image_encoder(flatten_input['image'] / 255.0)
            x_extra = self.extra_encoder(flatten_input)
            x = th.cat([x_img, x_extra], dim=1)

        assert not self.is_sequential, "no RRN in my multi-head-attention network!"

        x = x.view(*batch_shape, *x.shape[1:])
        x = self.attention_layer(x)

        return self.critic_linear(x), x, rnn_hxs