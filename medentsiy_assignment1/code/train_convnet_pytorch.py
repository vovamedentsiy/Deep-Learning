"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  ind_pred = np.argmax(predictions.detach().numpy(), axis=1)
  ind_targets = np.argmax(targets.detach().numpy(), axis=1)
  accuracy = (ind_pred == ind_targets).mean()

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # will be used to compute accuracy and loss for the whole train and test sets
  batch_size_acc = 500
  data_accuracy_loss = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)
  X_train_acc, y_train_acc = data_accuracy_loss['train'].images, data_accuracy_loss['train'].labels
  X_test_acc, y_test_acc = data_accuracy_loss['test'].images, data_accuracy_loss['test'].labels
  steps_train = int(X_train_acc.shape[0] / batch_size_acc)
  steps_test = int(X_test_acc.shape[0] / batch_size_acc)
  #


  data = cifar10_utils.get_cifar10(data_dir = FLAGS.data_dir)

  n_classes = data['train'].labels.shape[1]
  n_inputs = data['train'].images.shape[1]*data['train'].images.shape[2]*data['train'].images.shape[3]
  batch_size = FLAGS.batch_size
  m_steps = FLAGS.max_steps
  alpha = FLAGS.learning_rate



  cnn = ConvNet(3, 10)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(cnn.parameters(), lr=alpha)


  x_ax = []
  acc_train = []
  acc_test = []
  loss_train = []
  loss_test = []

  for step in range(m_steps):

        x, y = data['train'].next_batch(batch_size)
        n = x.shape
        x = torch.from_numpy(x)

        y_pred = cnn(x)
        labels = torch.LongTensor(y)

        loss = criterion(y_pred, torch.max(labels, 1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if step % FLAGS.eval_freq == 0:
            print('Iteration ',step)

            x_ax.append(step)


            acc_ = []
            loss_ = []
            for i in range(steps_train):
                x_acc = X_train_acc[i * batch_size_acc:(i + 1) * batch_size_acc]
                y_acc = y_train_acc[i * batch_size_acc:(i + 1) * batch_size_acc]
                x_acc = torch.from_numpy(x_acc)
                y_acc = torch.LongTensor(y_acc)

                y_pred = cnn.forward(x_acc)
                acc_.append(accuracy(y_pred, y_acc))
                loss_.append(float(criterion(y_pred, torch.max(y_acc, 1)[1])))

            acc_train.append(np.mean(acc_))
            loss_train.append(np.mean(loss_))

            acc_ = []
            loss_ = []
            for i in range(steps_test):
                x_acc = X_test_acc[i * batch_size_acc:(i + 1) * batch_size_acc]
                y_acc = y_test_acc[i * batch_size_acc:(i + 1) * batch_size_acc]
                x_acc = torch.from_numpy(x_acc)
                y_acc = torch.LongTensor(y_acc)

                y_pred = cnn.forward(x_acc)
                acc_.append(accuracy(y_pred, y_acc))
                loss_.append(float(criterion(y_pred, torch.max(y_acc, 1)[1])))

            acc_test.append(np.mean(acc_))
            loss_test.append(np.mean(loss_))


            print('Max train accuracy ', max(acc_train))
            print('Max test accuracy ', max(acc_test))
            print('Min train loss ', min(loss_train))
            print('Min test loss ', min(loss_test))


  x_ax = np.array(x_ax)
  acc_test = np.array(acc_test)
  acc_train = np.array(acc_train)
  loss_test = np.array(loss_test)
  loss_train = np.array(loss_train)

  print('Max train accuracy ', max(acc_train))
  print('Max test accuracy ', max(acc_test))
  print('Min train loss ', min(loss_train))
  print('Min test loss ', min(loss_test))

  fig = plt.figure()
  ax = plt.axes()

  plt.title("CNN Pytorch. Accuracy curves")
  ax.plot(x_ax, acc_train, label='train');
  ax.plot(x_ax, acc_test, label='test');
  ax.set_xlabel('Step');
  ax.set_ylabel('Accuracy');
  plt.legend();
  plt.savefig('accuracy_cnn.jpg')

  fig = plt.figure()
  ax = plt.axes()
  plt.title("CNN Pytorch. Loss curves")
  ax.plot(x_ax, loss_train, label='train');
  ax.plot(x_ax, loss_test, label='train');
  ax.set_xlabel('Step');
  ax.set_ylabel('Loss');
  ax.set_ylim(top=10, bottom=0)
  plt.legend();
  plt.savefig('loss_cnn.jpg')



  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
