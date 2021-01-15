# Encoder layer and Composite encoder stack for transformer model.

from copy import deepcopy

import torch
import torch.nn as nn

from transformer_block.attention import MultiHeadedAttention
from transformer_block.sublayers import AddAndNormWithDropoutLayer,PositionWiseFFNLayer

class EncoderBlock(nn.Module):

