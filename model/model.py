import torch.nn as nn
from utils import fix_random_seed_as, STEFunction
import torch
import math
import torch.nn.functional as F
import numpy as np
import pickle
import scipy.sparse as sp



class TRACEModel(nn.Module):
    def __init__(self, num_items, num_items_l, args) -> None:
        super().__init__()
        self.trace = TRACE(num_items, args.seed, args.dataset)
        self.out = nn.Linear(self.trace.hidden, num_items_l + 1)

    def forward(self, x, time):
        x = self.trace(x, time)
        return self.out(x)
    
class TRACE(nn.Module):
    def __init__(self, num_items, seed, dataset):
        super().__init__()

        fix_random_seed_as(seed)

        max_len = 100
        n_layers = 3
        heads = 4
        vocab_size = num_items + 2
        hidden = 256
        self.hidden = hidden
        dropout = 0.1

        self.embedding = EmbeddingLayer(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout, dataset=dataset)

        self.sample1 = nn.Linear(self.hidden, 128)
        self.sample2 = nn.Linear(128,1)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout, max_len) for _ in range(n_layers)])



    def forward(self, x, times):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x, time_embedding = self.embedding(x, times)
        
        pmask = None

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask, pmask, time_embedding)
        
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, dataset="mimic4"):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.time_embedding = TimeEmbedding(embed_size)
    def forward(self, sequence, times):

        time_feature = self.time_embedding(times)
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x), self.dropout(time_feature)

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, max_len):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout, max_len=max_len)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, pmask, time):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask, pmask = pmask, time = time))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, max_len=256):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(max_len)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, pmask = None, time = None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key1, key2, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, time, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key1, value, mask=mask, dropout=self.dropout, pmask = pmask, time = key2)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)

class Attention(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.mask = DifferentiableMask((max_len, max_len))
        self.time_decay_coefficient = nn.Parameter(torch.tensor(1.0))
        self.tanh = torch.nn.Tanh()

    def forward(self, query, key, value, mask=None, dropout=None, pmask = None, time = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if time is not None:
            time_score = torch.matmul(query, time.transpose(-2,-1)) / math.sqrt(query.size(-1))
            scores +=  time_score
        p_attn = F.softmax(scores, dim=-1)
        p_attn = p_attn.mul(self.mask())

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class DifferentiableMask(nn.Module):
    def __init__(self, size, init_value=3.0):
        super(DifferentiableMask, self).__init__()
        self.size = size
        self.mask_logits = nn.Parameter(torch.full(size, init_value))
        self.random_init()
    def random_init(self):
        self.mask_logits = torch.nn.init.uniform_(self.mask_logits, a=2, b=5)
    def forward(self):
        return torch.sigmoid(self.mask_logits)
    
class TimeEmbedding(nn.Module):
    def __init__(self, embed_size=256):
        super(TimeEmbedding, self).__init__()
        self.tanh = nn.Tanh()
        self.selection_layer = torch.nn.Linear(1, 64)
        self.time_layer = torch.nn.Linear(64, embed_size)
        self.periodic_layer = torch.nn.Linear(2, embed_size)
        
    def forward(self, times):
         # periodical encoding
        time_sin = torch.sin(2 * math.pi * times / 24)
        time_cos = torch.cos(2 * math.pi * times / 24)
        time_periodic = torch.stack([time_sin, time_cos], dim=2)  
        time_periodic = self.periodic_layer(time_periodic) 

        # decay encoding
        times = times.unsqueeze(2) / 180 
        times = self.selection_layer(times)
        time_feature = 1 - self.tanh(torch.pow(times, 2))
        time_feature = self.time_layer(time_feature)

        time_embedding = time_feature + time_periodic  

        return time_embedding