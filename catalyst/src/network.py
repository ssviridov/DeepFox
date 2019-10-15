from typing import List
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils
from .resnet import FixupResNet


class StateEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 history_lengths: List = None,
                 obs: List = None,
                 image_dim: int = 512
                 ):
        super(StateEncoder, self).__init__()

        assert len(history_lengths) == len(obs), "Not all history lengths for observations are specified!"

        self.in_channels = in_channels
        self.obs_to_hist = dict(zip(obs, history_lengths))
        self.image_dim = image_dim

        self.image_encoder = FixupResNet(self.in_channels * self.obs_to_hist['image'], self.image_dim)

    def concat_vector_obs(self, x, bs):
        return torch.cat([x[k][:, -self.obs_to_hist[k]:, :].view(bs, -1) for k in x.keys() if k != "image"], dim=1)

    def forward(self, x):

        image = x['image'] / 255.
        batch_size, history_len, c, h, w = image.shape
        image = image.view(batch_size, -1, h, w)

        x_img = self.image_encoder(image)
        vector_obs = self.concat_vector_obs(x, batch_size)
        return torch.cat([x_img, vector_obs], dim=1)


def _get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    features = features or [64, 128, 64]
    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**linear_params):
        layers = [nn.Linear(**linear_params)]
        if use_normalization:
            layers.append(nn.LayerNorm(linear_params["out_features"]))
        if use_dropout:
            layers.append(nn.Dropout(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    features.insert(0, history_len * in_features)
    params = []
    for i, (in_features, out_features) in enumerate(utils.pairwise(features)):
        params.append(
            {
                "in_features": in_features,
                "out_features": out_features,
                "bias": use_bias,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    return net


class StateNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        observation_net: nn.Module = None,
        aggregation_net: nn.Module = None,
    ):
        """
        Abstract network, that takes some tensor
        T of shape [bs; history_len; ...]
        and outputs some representation tensor R
        of shape [bs; representation_size]

        input_T [bs; history_len; in_features]

        -> observation_net (aka observation_encoder) ->

        observations_representations [bs; history_len; obs_features]

        -> aggregation_net (flatten in simplified case) ->

        aggregated_representation [bs; hid_features]

        -> main_net ->

        output_T [bs; representation_size]

        Args:
            main_net:
            observation_net:
            aggregation_net:
        """
        super().__init__()
        self.main_net = main_net
        self.observation_net = observation_net
        self.aggregation_net = aggregation_net

    def forward(self, state):
        x = self.observation_net(state)
        x = self.main_net(x)
        return x

    @classmethod
    def get_from_params(
        cls,
        observation_net_params=None,
        # aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":

        observation_net_params = deepcopy(observation_net_params)
        main_net_params = deepcopy(main_net_params)

        observation_net = StateEncoder(**observation_net_params)

        aggregation_net = None

        main_net = _get_linear_net(**main_net_params)

        net = cls(
            observation_net=observation_net,
            aggregation_net=aggregation_net,
            main_net=main_net
        )

        return net
