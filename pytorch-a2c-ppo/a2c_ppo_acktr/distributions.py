import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import AddBias, init, default_init, nonlinearities

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()

class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_sizes=tuple(), nl=nn.Tanh):
        super(Categorical, self).__init__()
        self._build_network(num_inputs, num_outputs, hidden_sizes, nl)

    def _build_network(self, num_inputs, num_outputs, hidden_sizes, nl):
        nl_module = nonlinearities[nl]

        layers = []
        prev_dim = num_inputs
        for i, h in enumerate(hidden_sizes):
            #fc_name = 'policy_fc{}'.format(i + 1)
            #nl_name = 'policy_{}_{}'.format(nl, i+1)
            layers.append( default_init(nn.Linear(prev_dim, h), nl) )
            layers.append( nl_module() )
            prev_dim = h

        #fc_name = 'policy_fc{}'.format(len(hidden_sizes)+1)
        layers.append(default_init(nn.Linear(prev_dim, num_outputs), 0.01))
        if len(layers)>1:
            self.policy_head = nn.Sequential(*layers)
        else:
            self.policy_head = layers[0]

    def forward(self, x):
        x = self.policy_head(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
