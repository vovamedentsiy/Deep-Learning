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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device_name = device

        self.bias_h = nn.Parameter(torch.zeros(num_hidden, device = self.device_name), requires_grad=True)
        self.bias_p = nn.Parameter(torch.zeros(num_classes, device = self.device_name), requires_grad=True)

        self.W_ph = nn.Parameter(0.0001*torch.randn(num_hidden, num_classes, device = self.device_name), requires_grad=True)

        self.W_hh = nn.Parameter(0.0001*torch.randn(num_hidden, num_hidden, device = self.device_name), requires_grad=True)
        self.W_hx = nn.Parameter(0.0001*torch.randn(input_dim, num_hidden, device = self.device_name), requires_grad=True)
        self.h = nn.Parameter(torch.zeros(self.batch_size, self.num_hidden, device = self.device_name), requires_grad=False)

    def forward(self, x):
        # Implementation here ...

        h = self.h
        for i in range(self.seq_length):

            if self.input_dim == 1:
                x_ = x[:, i].view(-1, 1).to(self.device_name)
            else:
                x_ = x[:, : , i].to(self.device_name)
            t1 = torch.mm(x_, self.W_hx)
            t2 = torch.mm(h, self.W_hh)
            h = torch.tanh(t1 + t2 + self.bias_h[None, :])

        p = torch.mm(h, self.W_ph)
        p = p + self.bias_p[None, :]

        return p
