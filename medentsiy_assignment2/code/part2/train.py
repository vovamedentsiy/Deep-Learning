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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#from part2.dataset import TextDataset
#from part2.model import TextGenerationModel

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def acc(predictions, targets):

    ind_pred = np.argmax(predictions.detach().cpu().numpy(), axis=1)
    ind_targets = targets.detach().cpu().numpy()
    acc = (ind_pred == ind_targets).mean()

    return acc

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    dataset = TextDataset(filename=config.txt_file, seq_length=config.seq_length)

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size = config.batch_size , seq_length = config.seq_length , vocabulary_size = dataset.vocab_size, lstm_num_hidden=config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers, device = config.device).to(config.device)  # fixme

    # Initialize the dataset and data loader (note the +1)

    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)  # fixme

    # if the number of required steps exceed the size of the data, then more than one epoch required and I need the outer loop for it
    steps_in_epoch = int(dataset.__len__()/config.batch_size)+1
    epochs = int(config.train_steps/steps_in_epoch) + 1
    print('EPOCHS ', epochs)
    print('STEPS IN EPOCH ', steps_in_epoch)
    print('TOTAL NUMBER OF STEPS  ', config.train_steps)
    #print('MAX POSSIBLE NUMBER OF STEPS  ', dataset.__len__(), '  TOTAL NUMBER OF STEPS  ', config.train_steps)

    #save_model and save_model1 are lists with the number of steps for which I save the model
    save_model = [int(h*0.2*config.train_steps) for h in range(5)]
    save_model1 = [100, 500, 1500]
    accuracy_dict = {}
    loss_dict = {}

    for j in range(epochs):
      print('EPOCH ', j)

      for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################

        batch_targets = torch.stack(batch_targets).to(config.device)
        y_pred = model.forward(batch_inputs).transpose(0, 2)

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        batch_targets = batch_targets.transpose(0, 1)
        loss = criterion(y_pred, batch_targets)
        loss.backward()
        optimizer.step()
        accuracy = acc(y_pred, batch_targets)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if int(step +j*steps_in_epoch) % config.print_every == 0:

            accuracy_dict[int(step + j * steps_in_epoch)] = accuracy
            loss_dict[int(step + j * steps_in_epoch)] = float(loss)

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), int(step +j*steps_in_epoch),
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.sample_every:
            # Generate some sentences by sampling from the model
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

        if int(step +j*steps_in_epoch) in save_model:
                name_model = 'model_'+str(int(step +j*steps_in_epoch))+'.pickle'
                torch.save(model.state_dict(), name_model)

        if int(step +j*steps_in_epoch) in save_model1:
                name_model = 'model_'+str(int(step +j*steps_in_epoch))+'.pickle'
                torch.save(model.state_dict(), name_model)

    torch.save(model.state_dict(), 'model_final.pickle')
    f1 = open("accuracy.txt","w")
    f1.write( str(accuracy_dict) )
    f1.close()

    f2 = open("loss.txt","w")
    f2.write( str(loss_dict) )
    f2.close()

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on") #default = 'assets/book_EN_democracy_in_the_US.txt')#
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cpu", help=" 'cpu' or 'cuda:0' ")

    config = parser.parse_args()

    # Train the model
    train(config)
