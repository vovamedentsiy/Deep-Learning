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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

#from part1.dataset import PalindromeDataset
#from part1.vanilla_rnn import VanillaRNN
#from part1.lstm import LSTM

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def acc(predictions, targets):

    ind_pred = np.argmax(predictions.detach().cpu().numpy(), axis=1)
    ind_targets = targets.detach().cpu().numpy()
    acc = (ind_pred == ind_targets).mean()

    return acc

def train(config):

    assert config.model_type in ('RNN', 'LSTM')


    # Initialize the model that we are going to use
    device = torch.device(config.device)  # fixme

    if config.model_type == 'RNN':
        model = VanillaRNN(seq_length = config.input_length, input_dim = config.input_dim, num_hidden = config.num_hidden, num_classes = config.num_classes, batch_size = config.batch_size, device=config.device).to(config.device)  # fixme
    elif config.model_type == 'LSTM':
        model = LSTM(seq_length = config.input_length, input_dim = config.input_dim, num_hidden = config.num_hidden, num_classes = config.num_classes, batch_size = config.batch_size, device=config.device).to(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.RMSprop(model.parameters(), lr = config.learning_rate) # fixme

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...

        ############################################################################
        # QUESTION: what happens here and why?
        """This function is used to bound the norm of the gradient within certain threshold (specified by the max\_norm argument),
        this technique can help with the problem of exploding gradients and makes training stable
        (avoid too large steps when update parameters )."""
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Add more code here ...
        batch_targets = batch_targets.to(config.device)

        y_pred = model.forward(batch_inputs)
        loss = criterion(y_pred, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = acc(y_pred, batch_targets)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    acc_test = []
    for i in range(10):
        (tr, te) = next(iter(data_loader))
        y_pred = model.forward(tr)
        acc_test.append(acc(y_pred, te))
    print('FINAL TEST ACCURACY: ', np.mean(acc_test))

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default='cuda:0', help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)