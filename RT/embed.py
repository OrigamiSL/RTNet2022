import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, group=1):
        super(TokenEmbedding, self).__init__()
        padding = 0
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, groups=group,
                                   kernel_size=1, padding=padding)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, group=1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, group=group)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)
