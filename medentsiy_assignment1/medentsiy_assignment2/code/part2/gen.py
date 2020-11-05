#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:53:26 2019

"""
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import TextDataset
from model import TextGenerationModel

def softmax(x):

    m = np.max(x, axis = 0, keepdims = True)
    out = np.exp(x - m)
    out = out/out.sum(axis = 0, keepdims = True)

    return out

def generate_greedy(model, dataset, sentence_len = 30):

    h1 = torch.zeros(model.lstm_num_layers,1, model.lstm_num_hidden).to(device=model.device_name)
    h2 = torch.zeros(model.lstm_num_layers, 1, model.lstm_num_hidden).to(device=model.device_name)

    chars = []
    random_char = torch.tensor([random.choice(list(dataset._ix_to_char.keys()))])
    chars.append(random_char)

    for i in range(sentence_len):
        y_pred, (h1, h2) = model.generate([chars[-1]], h1, h2)
        chars.append(torch.max(y_pred , 1)[1][y_pred.size(0) - 1])
    chars = [int(char.cpu().numpy()[0]) for char in chars]

    return dataset.convert_to_string(chars)

def generate_temperature(model, dataset, sentence_len = 30, T = 1):

    h1 = torch.zeros(model.lstm_num_layers,1, model.lstm_num_hidden).to(device=model.device_name)
    h2 = torch.zeros(model.lstm_num_layers, 1, model.lstm_num_hidden).to(device=model.device_name)

    chars = []
    random_char = torch.tensor([random.choice(list(dataset._ix_to_char.keys()))])
    chars.append(random_char)
    count = 0

    for i in range(sentence_len):
        y_pred, (h1, h2) = model.generate([chars[-1]], h1, h2)
        s = np.ravel(y_pred[0].detach().numpy())
        s = s/T
        prob = softmax(s)
        t_choice = torch.tensor(np.random.choice(dataset._vocab_size, 1, p=prob))
        greedy_choice = torch.max(y_pred , 1)[1][y_pred.size(0) - 1]
        count += int((greedy_choice == t_choice))
        chars.append(t_choice)
    chars = [int(char.cpu().numpy()[0]) for char in chars]

    #print(count, ' out of ', sentence_len,' coincide')
    return dataset.convert_to_string(chars)

def generate_temperature_given(model, dataset, sentence_len = 30, T = 1, given_sentence = 'Sleeping beauty is '):

    chars = [torch.tensor(dataset._char_to_ix[symbol])[None] for symbol in given_sentence]

    h1 = torch.zeros(model.lstm_num_layers, 1, model.lstm_num_hidden).to(device=model.device_name)
    h2 = torch.zeros(model.lstm_num_layers, 1, model.lstm_num_hidden).to(device=model.device_name)
    y_pred, (h1, h2) = model.generate(chars, h1, h2)
    chars.append(torch.max(y_pred , 1)[1][y_pred.size(0) - 1])

    count = 0

    for i in range(sentence_len):
        y_pred, (h1, h2) = model.generate([chars[-1]], h1, h2)
        s = np.ravel(y_pred[0].detach().numpy())
        s = s/T
        prob = softmax(s)
        t_choice = torch.tensor(np.random.choice(dataset._vocab_size, 1, p=prob))
        greedy_choice = torch.max(y_pred , 1)[1][y_pred.size(0) - 1]
        count += int((greedy_choice == t_choice))
        chars.append(t_choice)
    chars = [int(char.cpu().numpy()[0]) for char in chars]

    #print(count, ' out of ', sentence_len,' coincide')
    return dataset.convert_to_string(chars)

def generate_greedy_given(model, dataset, sentence_len = 30, given_sentence = 'Sleeping beauty is '):

    chars = [torch.tensor(dataset._char_to_ix[symbol])[None] for symbol in given_sentence]

    h1 = torch.zeros(model.lstm_num_layers, 1, model.lstm_num_hidden).to(device=model.device_name)
    h2 = torch.zeros(model.lstm_num_layers, 1, model.lstm_num_hidden).to(device=model.device_name)
    y_pred, (h1, h2) = model.generate(chars, h1, h2)
    chars.append(torch.max(y_pred , 1)[1][y_pred.size(0) - 1])
    for i in range(sentence_len):
        y_pred, (h1, h2) = model.generate([chars[-1]], h1, h2)
        chars.append(torch.max(y_pred , 1)[1][y_pred.size(0) - 1])
    chars = [int(char.cpu().numpy()[0]) for char in chars]

    return dataset.convert_to_string(chars)

batch_size = 64
seq_length = 30
vocabulary_size = 87
lstm_num_hidden=128
lstm_num_layers=2
device = 'cpu'


model = TextGenerationModel(batch_size, seq_length, vocabulary_size, lstm_num_hidden, lstm_num_layers, device ).to(device)
model.load_state_dict(torch.load('results_grim/model_final.pickle', map_location='cpu'))

dataset = TextDataset(filename='assets/book_EN_grimms_fairy_tails.txt', seq_length=30)

print(generate_greedy(model, dataset, 30))
print(generate_temperature(model, dataset, 30, 2))
print(generate_greedy_given(model, dataset, 180))
print(generate_temperature_given(model, dataset, 180, T = 0.5))
