from collections import OrderedDict

from a2c_ppo_acktr.model import CNNBase, NNBase, Bernoulli,\
    Categorical, DiagGaussian, MLPBase, Flatten, Policy
import gym
from torch import nn
import torch as th
from a2c_ppo_acktr.utils import init, conv_output_shape
import numpy as np
from .aai_layers import NaiveHistoryAttention, TemporalAttentionPooling

class AAIPolicy(Policy):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__() #each we skip Policy's initialization!
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = AAIBase

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


class AAIBase(NNBase):

    def __init__(
            self,
            obs_space,
            extra_obs=None,
            recurrent=False,
            hidden_size=512,
            extra_encoder_dim=256,
            image_encoder_dim=512,
    ):
        self._image_encoder_dim = image_encoder_dim
        #if there is no extra_obs then we don't need an extra_encoder!
        self._extra_encoder_dim = extra_encoder_dim if extra_obs else 0
        self._total_encoder_dim = self._image_encoder_dim + self._extra_encoder_dim
        #if recurrent is False there will be no layer after encoders ouputs:
        self._hidden_size = hidden_size if recurrent else self._total_encoder_dim
        self._extra_obs = extra_obs

        super(AAIBase, self).__init__(
            recurrent,
            self._image_encoder_dim+self._extra_encoder_dim,
            self._hidden_size
        )

        self.obs_shapes = self._get_obs_shapes(obs_space)
        self._image_only_obs = isinstance(obs_space, gym.spaces.Box)

        self.image_encoder = self._create_image_encoder()

        if self._extra_obs:
            self.extra_encoder = self._create_extra_encoder()
        else:
            self.extra_encoder = None

        self.critic_linear = self._create_critic()

        self.train()

    def _get_obs_shapes(self, obs_space):
        if isinstance(obs_space, gym.spaces.Dict):
            return {k:v.shape for k,v in obs_space.spaces.items()}
        else:
            return {"image":obs_space.shape}

    def _create_critic(self):
        init_ = lambda m:init(m, nn.init.orthogonal_, lambda x:nn.init.constant_(x, 0))
        return init_(nn.Linear(self._hidden_size, 1))

    def _create_image_encoder(self):
        num_channels = self.obs_shapes['image'][0]
        init_ = lambda m:init(m, nn.init.orthogonal_, lambda x:nn.init.
                              constant_(x, 0), nn.init.calculate_gain('relu'))

        return nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, self._image_encoder_dim)), nn.ReLU()
        )

    def _create_extra_encoder(self):
        if self._extra_obs:
            n_features = 0
            for k in self._extra_obs:
                k_shape = self.obs_shapes[k]
                assert len(k_shape) == 1, 'We account only for one dimensional extra obs'
                n_features += k_shape[0]

            init_ = lambda m:init(m, nn.init.orthogonal_, lambda x:nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))
            return nn.Sequential(
                init_(nn.Linear(n_features, self._extra_encoder_dim)),
                nn.ReLU()
            )
        else:
            return None

    def forward(self, input, rnn_hxs, masks, **kwargs):

        if self._image_only_obs:
            x = self.image_encoder(input/255.0)
        else:
            x_img = self.image_encoder(input['image']/255.0)
            inp_extra = th.cat([input[k] for k in self._extra_obs], dim=1)
            x_extra = self.extra_encoder(inp_extra)
            x = th.cat([x_img, x_extra], dim=1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class ImageVecMapBase(NNBase):

    def __init__(
            self,
            obs_space,
            extra_obs=None,
            policy="ff",
            hidden_size=512,
            extra_encoder_dim=512,
            image_encoder_dim=512,
    ):
        self._image_encoder_dim = image_encoder_dim
        # if there is no extra_obs then we don't need an extra_encoder!
        self._extra_encoder_dim = extra_encoder_dim if extra_obs else 0
        self._total_encoder_dim = self._image_encoder_dim + self._extra_encoder_dim
        # if recurrent is False there will be no layer after encoders ouputs:
        self._hidden_size = hidden_size if policy!='ff' else self._total_encoder_dim
        self._extra_obs = extra_obs
        self._policy_type = policy

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
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self._hidden_size, 1))

    def _create_image_encoder(self):
        num_channels = self.obs_shapes['image'][-3]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, self._image_encoder_dim)), nn.ReLU()
        )

    def _create_map_encoder(self):
        num_channels, H, W = self.obs_shapes['visited'][-3:]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(num_channels, 16, 4, 2)), nn.ReLU(),
            init_(nn.Conv2d(16, 32, 3, 1)), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten()
        )

    def _create_extra_encoder(self):
        input_features = 0
        if 'visited' in self._extra_obs:
            self._create_map_encoder()
            input_features += conv_output_shape(
                self.obs_shapes['visited'][-3:],
                self.map_encoder
            )[0]
        else:
            self.map_encoder = None
        # all images has at least 3 dims,
        # vectors would have 2(if stacked along time) or 1
        self._vector_obs = [k for k in self._extra_obs if len(self.obs_shapes[k]) < 3]
        for k in self._vector_obs:
            input_features += self.obs_shapes[k][-1]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.extra_linear = nn.Sequential(
            init_(nn.Linear(input_features, self._extra_encoder_dim)),
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

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


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
        self.attention_layer = TemporalAttentionPooling(self._total_encoder_dim)
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

        assert not self.is_recurrent, "no RRN in my multi-head-attention network!"

        x = x.view(*batch_shape, *x.shape[1:])
        x = self.attention_layer(x)

        return self.critic_linear(x), x, rnn_hxs


import torchvision
class AAIResnet(AAIBase):

    def __init__(self, *args, **kwargs):
        if "freeze_resnet" in kwargs:
            self.freeze_resnet = kwargs.pop('freeze_resnet')
        else:
            self.freeze_resnet = False

        super(AAIResnet, self).__init__(*args, **kwargs)

    def _create_image_encoder(self):
        num_channels = self.obs_shapes['image'][0]

        net = torchvision.models.resnet18(pretrained=self.freeze_resnet)

        resnet = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False,
                         count_include_pad=True))

        if self.freeze_resnet:
            for p in resnet.parameters():
                p.requires_grad = False
        else:
            resnet[0] = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        return resnet

    def forward(self, input, rnn_hxs, masks, **kwargs):

        if self._image_only_obs:
            x = self.image_encoder(input/255.0)
            x = x.view(x.shape[0],-1)
        else:
            x_img = self.image_encoder(input['image']/255.0)
            x_img = x_img.view(x_img.shape[0], -1)
            inp_extra = th.cat([input[k] for k in self._extra_obs], dim=1)
            x_extra = self.extra_encoder(inp_extra)
            x = th.cat([x_img, x_extra], dim=1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs