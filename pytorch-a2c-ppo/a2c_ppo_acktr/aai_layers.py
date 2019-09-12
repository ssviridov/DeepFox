import torch as th
from torch import nn


class NaiveHistoryAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(NaiveHistoryAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, memory_window):
        "Input should have the shape (batch_size, time_steps, embed_dim)"
        query = memory_window[:,-1].unsqueeze(0) #1, batch_size, embed_dim
        keys = memory_window[:,:-1].transpose(0,1) #time_steps-1, batch_size, embed_dim)
        attn_mem, attn_weights = self.mha(query, keys, keys)
        result = th.cat((query, attn_mem),dim=2).squeeze(0)
        return result #batch_size, 2*embedding dim


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

