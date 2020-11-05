# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device_name = device

        self.emb = nn.Embedding(vocabulary_size, vocabulary_size)
        self.emb.weight.requires_grad = False
        self.emb.weight.data = torch.eye(vocabulary_size)

        self.layer1  = nn.LSTM(input_size = vocabulary_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers)
        self.layer2 = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x):
        # Implementation here...

        x = torch.stack(x).to(self.device_name)
        bs = x.size(1)
        self.h1 = torch.zeros(self.lstm_num_layers, bs, self.lstm_num_hidden).to(device=self.device_name)
        self.h2 = torch.zeros(self.lstm_num_layers, bs, self.lstm_num_hidden).to(device=self.device_name)
        x_ = self.emb(x)
        x_, (self.h1, self.h2) = self.layer1(x_, (self.h1, self.h2))
        out = self.layer2(x_).transpose(1,2)

        return out

    def generate(self, x, h1, h2):
        '''function used to generate new sentences char by char'''

        x = torch.stack(x).to(self.device_name)
        x_ = self.emb(x)
        x_, (h1, h2) = self.layer1(x_, (h1, h2))
        out = self.layer2(x_).transpose(1,2)

        return out, (h1, h2)
