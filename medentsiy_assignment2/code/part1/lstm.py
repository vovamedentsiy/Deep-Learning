################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device_name = device


        self.bias_g = nn.Parameter(torch.zeros(num_hidden, device = self.device_name), requires_grad=True)
        self.bias_i = nn.Parameter(torch.zeros(num_hidden,device = self.device_name), requires_grad=True)
        self.bias_f = nn.Parameter(torch.zeros(num_hidden, device = self.device_name), requires_grad=True)
        self.bias_o = nn.Parameter(torch.zeros(num_hidden, device = self.device_name), requires_grad=True)
        self.bias_p = nn.Parameter(torch.zeros(num_classes, device = self.device_name), requires_grad=True)



        self.W_gh = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(num_hidden,  num_hidden, device=self.device_name)),
            requires_grad=True)

        self.W_ih = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(num_hidden, num_hidden, device=self.device_name)),
            requires_grad=True)

        self.W_fh = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(num_hidden, num_hidden, device=self.device_name)),
            requires_grad=True)

        self.W_oh = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(num_hidden, num_hidden, device=self.device_name)),
            requires_grad=True)

        self.W_gx = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(input_dim, num_hidden, device=self.device_name)),
            requires_grad=True)
        self.W_ix = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(input_dim, num_hidden, device=self.device_name)),
            requires_grad=True)
        self.W_fx = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(input_dim, num_hidden, device=self.device_name)),
            requires_grad=True)
        self.W_ox = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(input_dim, num_hidden, device=self.device_name)),
            requires_grad=True)

        self.W_ph = nn.Parameter(
            torch.nn.init.xavier_normal(torch.randn(num_hidden, num_classes, device=self.device_name)),
            requires_grad=True)

        self.c = nn.Parameter(torch.zeros(self.batch_size, self.num_hidden, device = self.device_name), requires_grad=False)
        self.h = nn.Parameter(torch.zeros(self.batch_size, self.num_hidden, device = self.device_name), requires_grad=False)

    def forward(self, x):
        # Implementation here ...
        c = self.c
        h = self.h

        for i in range(self.seq_length):
            if self.input_dim == 1:
                x_ = x[:, i].view(-1, 1).to(self.device_name)
            else:
                x_ = x[:, : , i].to(self.device_name)

            g = torch.tanh(torch.mm(x_, self.W_gx) + torch.mm(h, self.W_gh) + self.bias_g[None, :])
            i = torch.sigmoid(torch.mm(x_, self.W_ix) + torch.mm(h, self.W_ih) + self.bias_i[None, :])
            f = torch.sigmoid(torch.mm(x_, self.W_fx) + torch.mm(h, self.W_fh) + self.bias_f[None, :])
            o = torch.sigmoid(torch.mm(x_, self.W_ox) + torch.mm(h, self.W_oh) + self.bias_o[None, :])

            c = torch.mul(g, i) + torch.mul(c, f)
            h = torch.mul(torch.tanh(c), o)
        p = torch.mm(h, self.W_ph) + self.bias_p[None, :]

        return p