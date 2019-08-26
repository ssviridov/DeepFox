import torch as th
from torch import nn


class HistoryAttention(nn.Module):
    "IN_PROGRESS..."
    def __init__(self, embed_dim, num_heads, history_size):
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.history_size = history_size


if __name__ == "__main__":
    batch_size = 1
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

