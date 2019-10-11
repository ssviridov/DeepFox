import torch as th
from torch import nn
from collections import deque

def outer_init(layer: nn.Module) -> None:
    """
    Initialization for output layers of policy and value networks typically
    used in deep reinforcement learning literature.
    """
    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        v = 3e-3
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)

class Temperature(nn.Module):

    def __init__(self):
        super(Temperature, self).__init__()

    def forward(self, x, t=1.):
        return x/t


class TemporalAttentionPooling(nn.Module):
    """Unashamedly got this from Catalyst framework:)"""
    name2activation = {
        "softmax": nn.Softmax(dim=1),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    def __init__(self, features_in, activation=None, kernel_size=1, **params):
        super().__init__()
        self.features_in = features_in
        activation = activation or "softmax"

        self.attention_pooling = nn.Sequential(
            nn.Conv1d(
                in_channels=features_in,
                out_channels=1,
                kernel_size=kernel_size,
                **params
            ),
            TemporalAttentionPooling.name2activation[activation]
        )
        self.attention_pooling.apply(outer_init)

    def forward(self, curr_obs, memory_window):
        """
        :param features: [batch_size, history_len, feature_size]
        :return:
        """
        x = memory_window
        #x = features[:,:-1,:]
        #x_last = features[:,-1,:]
        batch_size, history_len, feature_size = x.shape

        x = x.view(batch_size, history_len, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attention_pooling(x_a) * x_a).transpose(1, 2)
        x_attn = x_attn.sum(1, keepdim=True)

        result = th.cat([x_attn.squeeze(1), curr_obs], dim=-1)
        return result


class NaiveHistoryAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(NaiveHistoryAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, cur_obs, memory_window):
        "Input should have the shape (batch_size, time_steps, embed_dim)"
        #query = memory_window[:,-1].unsqueeze(0) #1, batch_size, embed_dim
        #keys = memory_window[:,:-1].transpose(0,1) #time_steps-1, batch_size, embed_dim)
        query = cur_obs.unsqueeze(0)
        keys = memory_window.transpose(0,1)
        attn_mem, attn_weights = self.mha(query, keys, keys)
        result = th.cat((query, attn_mem),dim=2).squeeze(0)
        return result #batch_size, 2*embedding dim


class CachedAttention(nn.Module):

    def __init__(self, attn_module, memory_len):
        super(CachedAttention, self).__init__()
        self.attn_module = attn_module
        self.memory_len = memory_len

    def forward(self, input, attn_cache, mask):
        #history_window = th.cat(tuple(attn_cache), dim=1)
        attn_cache = attn_cache * mask.view(-1,1,1)
        result = self.attn_module(input, attn_cache)
        attn_cache = th.cat([attn_cache[:,1:], input.unsqueeze(1)], dim=1)
        return result, attn_cache

if __name__ == "__main__":
    batch_size = 2
    L_trg = 1
    L_src = 8
    embed_size = 10
    n_heads = 2

    mha = nn.MultiheadAttention(embed_size, n_heads)

    query = th.randn(L_trg, batch_size, embed_size)
    memory = th.randn(L_src, batch_size, embed_size)

    print("query size:", query.shape)
    print("memory size:", memory.shape)
    result, attn_weights = mha(query, memory, memory)
    print("result:", result.shape) #trg_len, batch_size, embed_dim
    print("attn_weights:", attn_weights.shape) #batch_size, trg_len, src_len
    print("attn_weights:", attn_weights.squeeze().tolist())

