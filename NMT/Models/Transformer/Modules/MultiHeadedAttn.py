import math
import torch
import torch.nn as nn
from torch.autograd import Variable



class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, model_dim, dropout=0.1):
        assert model_dim % num_heads == 0
        self.dim_per_head = model_dim // num_heads
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads

        self.linear_keys = nn.Linear(model_dim,
                                     num_heads * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       num_heads * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      num_heads * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            return x.view(batch_size, -1, num_heads, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, num_heads * dim_per_head)

        # 1) Project key, value, and query.
        key_up = shape(self.linear_keys(key))
        value_up = shape(self.linear_values(value))
        query_up = shape(self.linear_query(query))

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(Variable(mask), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)

        top_attn = attn \
            .view(batch_size, num_heads,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()
        
        return output, top_attn