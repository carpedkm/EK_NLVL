import torch
import numpy as np
from torch import nn

import src.model.dynamic_filters as DF
import src.utils.pooling as POOLING

class DynamicFilter(nn.Module):
    def __init__(self, cfg):
        super(DynamicFilter, self).__init__()
        self.cfg = cfg

        factory = getattr(DF, 'LSTM')
        self.tail_df = factory(cfg)

        factory = getattr(POOLING, 'MeanPoolingLayer')
        self.pooling_layer = factory()

        factory = getattr(DF, "MLP")
        self.head_df = factory(cfg)

    def forward(self, sequences, lengths=None):
        # print('df execution')
        output, _ = self.tail_df(sequences, lengths)
        output = self.pooling_layer(output, lengths)
        output = self.head_df(output)
        return output, lengths 
