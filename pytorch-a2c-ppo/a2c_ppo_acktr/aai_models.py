from .model import CNNBase, NNBase, Bernoulli,\
    Categorical, DiagGaussian, MLPBase, Flatten, Policy
import gym
from torch import nn
import torch as th
from a2c_ppo_acktr.utils import init

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
        assert num_channels == 3, 'Pretrained resnet knows no history!'

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