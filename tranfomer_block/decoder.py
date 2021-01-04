# Decoder layer and the Composity decoder stack for the transformer model.

from copy import deepcopy
import torch
import torch.nn as nn

from transformer_block.attention import MultiHeadedAttetion
from transformer_block.sublayer import AddAndNormWithDropoutLayer, PositionWiseFFNLayer

class DecoderBlock(nn.Module):
    def __init__(self, size: int, self_attention: MultiHeadedAttetion, feed_forward: PositionWiseFFNLayer, dropout: float):
        super(DecoderBlock, self).__init__()

        self._size = size
        # self attention
        self._self_attention = self_attention
        # add + norm layer
        self._add_norm_layer_1 = AddAndNormWithDropoutLayer(size, dropout)
        # source-attention
        self._source_attention = source_attention
        # add + norm layer
        self._add_norm_layer_2 = AddAndNormWithDropoutLayer(size, dropout)
        # FFN
        self._feed_forward = feed_forward
        # add + norm layer
        self._add_norm_layer_3 = AddAndNormWithDropoutLayer(size, dropout)

    @property
    def size(self) -> int:
        return self._size

    def forward(self, values: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        values = self._add_norm_layer_1(values, lambda x: self._self_attention(x, x, x, tgt_mask))
        values = self._add_norm_layer_2(values, lambda x: self._source_attetion(x, memory, src_mask))
        values = self._add_norm_layer_3(values, self._feed_forward)
        return values

class CompositeDecoder(nn.Module):
    def __init__(self, layer: DecoderBlock, num_layers: int):
        super(CompositeDecoder, self).__init__()
        self._layers = nn.ModuleList([deepcopy(layer)] * num_layers)
        self._add_norm = nn.BatchNorm1d(layer.size, momentum=None, affine=False)

    def forward(self, values: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor,  tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            values = layer(values, memory, src_mask, tgt_mask)
        return self._add_norm(values)

