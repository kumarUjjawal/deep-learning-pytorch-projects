# Multi-Headed attention layer to be used in the transfomer class.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    
    def __init__(self, num_heads: int,
                 dim_model: int,
                 dropout: float,
                 residual_dropout: float = 0.0):
        super(MultiHeadedAttention, self).__init__()

        self._dim_head = dim_model // num_heads
        self._num_heads = num_heads

        # linear layers
        self._linear_layer_queries = nn.Linear(dim_model, dim_model)
        self._linear_layer_keys = nn.Linear(dim_model, dim_model)
        self._linear_layer_values = nn.Linear(dim_model, dim_model)
        self._linear_layer_final = nn.Linear(dim_model, dim_model)

        self._attention_tensor = None
        self.dropout = nn.Dropout(p=dropout)

        # GPT specific
        self._residual_dropout = nn.Dropout(p=residual_dropout)

    def compute_attention(self, query: torch.Tensor, 
                          key: torch.Tensor,
                          value: torch.Tensor,
                          mask: torch.Tensor):
        assert query.size(-1) == self._dim_head

        scores = (query @ key.transpose(2,3)) / math.sqrt(self._dim_head)

        scores = scores.masked_fill(mask == 0, -1.e9)

        attention_probas = self._dropout(F.softmax(scores, dim=1))

        attention_values = attention_probas @ value
        
        return attention_values, attention_probas

    def forward(self, query, key, value, mask):
        batch_size, max_seq_length, dim_model = query.size()
        assert self._num_heads * self._dim_head == dim_model

        query = self._linear_layer_queries(query).view(batch_size, max_seq_length, self._num_heads, self._dim_head).transpose(1,2)

        key = self._linear_layer_keys(key).view(batch_size, max_seq_lenght, self._num_heads, self._dim_head).transpose(1,2)

        value = self._linear_layer_values(value).view(batch_size, max_seq_lenght,, self._num_heads, self._dim_head).transpose(1,2)

        mask = mask.unsqueeze(1)

        attention_values, self._attention_tensor = self.compute_attention(query, key, value, mask)

        attention_values = attention_values.transpose(1,2).contiguous().view(batch_size, max_seq_lenght, self._num_heads * self._dim_head)

        attention_values = self._linear_layer_final(attention_values)

        attention_values = self._residual_dropout(attention_values)
        return attention_values

