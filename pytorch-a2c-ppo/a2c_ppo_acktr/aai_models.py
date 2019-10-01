from collections import OrderedDict

from a2c_ppo_acktr.model import CNNBase, NNBase, Bernoulli,\
    Categorical, DiagGaussian, MLPBase, Flatten, Policy
import gym
from torch import nn
import torch as th
from a2c_ppo_acktr.utils import init, conv_output_shape
import torch.nn.functional as F

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
            recurrent=False,
            hidden_size=512,
            extra_encoder_dim=512,
            image_encoder_dim=512,
    ):
        self._image_encoder_dim = image_encoder_dim
        # if there is no extra_obs then we don't need an extra_encoder!
        self._extra_encoder_dim = extra_encoder_dim if extra_obs else 0
        self._total_encoder_dim = self._image_encoder_dim + self._extra_encoder_dim
        # if recurrent is False there will be no layer after encoders ouputs:
        self._hidden_size = hidden_size if recurrent else self._total_encoder_dim
        self._extra_obs = extra_obs

        super(ImageVecMapBase, self).__init__(
            recurrent,
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
        num_channels = self.obs_shapes['image'][0]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, self._image_encoder_dim)), nn.ReLU()
        )

    def _create_map_encoder(self):
        num_channels, H, W = self.obs_shapes['visited']
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(num_channels, 16, 4, 2)), nn.ReLU(), #was num_channes -> 16
            init_(nn.Conv2d(16, 32, 3, 1)), nn.ReLU(), # was 16, -> 32
            nn.AvgPool2d(2, 2),#init_(nn.Conv2d(32, 64, 3, 1)), nn.ReLU(), # was AvgPool2D without relu
            Flatten()
        )

    def _create_extra_encoder(self):
        input_features = 0
        if 'visited' in self._extra_obs:
            self._create_map_encoder()
            input_features += conv_output_shape(self.obs_shapes['visited'], self.map_encoder)[0]
        else:
            self.map_encoder = None

        self._vector_obs = [k for k in self._extra_obs if len(self.obs_shapes[k]) == 1]
        for k in self._vector_obs:
            input_features += self.obs_shapes[k][0]

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

            encoder = OrderedDict([
                ("conv1", init_(nn.Conv2d(num_channels, 32, 3, 1))),
                ("relu1", nn.ReLU()),  # was num_channes -> 16
                ('conv2', init_(nn.Conv2d(32, 32, 3, 1))),
                ('relu2', nn.ReLU()),  # was 16, -> 32
                ('conv3', init_(nn.Conv2d(32, 16, 3, 1))),
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, num_channels, image_dim, block = FixupBasicBlock, layers = [2, 2, 2, 2], , ):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512 * block.expansion, image_dim)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2 / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv3.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x