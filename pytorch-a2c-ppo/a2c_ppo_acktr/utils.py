import glob
import os
import json
import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize

nonlinearities = {
        "relu": nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'tanh': nn.Tanh,
    }

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    if isinstance(gain, str):
        gain = nn.init.calculate_gain(gain)

    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def default_init(module, gain=1):
    return init(
        module, nn.init.orthogonal_,
        lambda x:nn.init.constant_(x, 0),
        gain
    )


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def conv_output_shape(input_shape, layers):
    with torch.no_grad():
        x = torch.randn(input_shape).unsqueeze(0)
        if isinstance(layers, dict):
            for k, l in layers.items():
                x = l(x)
            return x.shape[1:]

        return layers(x).shape[1:]
