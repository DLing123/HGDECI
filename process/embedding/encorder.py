#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2020/11/25 17:03
# @Author   : Dling
# @FileName : encorder.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim


class Encoder(nn.Module):
    def __init__(self, confs):
        nn.Module.__init__(self)

        self.max_length = confs["max_length"]
        self.hidden_size = confs["hidden_size"]
        self.embedding_dim = confs["word_embedding_dim"] + confs["pos_embedding_dim"] * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(confs["max_length"])

        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2) # n x hidden_size

    def pcnn(self, inputs, mask):
        x = self.conv(inputs.transpose(1, 2)) # n x hidden x length
        mask = 1 - self.mask_embedding(mask).transpose(1, 2) # n x 3 x length
        pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
        pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
        x = torch.cat([pool1, pool2, pool3], 1)
        x = x.squeeze(2) # n x (hidden_size * 3)
