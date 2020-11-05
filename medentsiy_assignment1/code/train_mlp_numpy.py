"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import matplotlib.pyplot as plt
from modules import *

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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

  ind_pred = np.argmax(predictions, axis=1)
  ind_targets = np.argmax(targets, axis=1)
  accuracy = (ind_pred == ind_targets).mean()

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  cifar10 = cifar10_utils.get_cifar10(data_dir = FLAGS.data_dir)
  alpha = FLAGS.learning_rate
  batch_size = FLAGS.batch_size
  n_classes = 10
  input_dim = 3*32*32

  mlp = MLP(input_dim, dnn_hidden_units, n_classes)

  loss = CrossEntropyModule()
  X_test, Y_test = cifar10['test'].images, cifar10['test'].labels
  X_test = np.reshape(X_test, (X_test.shape[0], -1))

  X_train, Y_train = cifar10['train'].images, cifar10['train'].labels
  X_train = np.reshape(X_train, (X_train.shape[0], -1))

  x_ax = []
  acc_train = []
  acc_test = []
  loss_print_tr = []
  loss_print_te = []

  for step in range(FLAGS.max_steps):

      x_train, y_train = cifar10['train'].next_batch(batch_size)
      x_train = np.reshape(x_train, (batch_size, -1))

      predictions = mlp.forward(x_train)

      loss_train = loss.forward(predictions, y_train)

      dout = loss.backward(predictions, y_train)
      mlp.backward(dout)

      for layer in mlp.layers:
        if isinstance(layer, LinearModule):
          layer.params['weight'] -= alpha * layer.grads['weight']
          layer.params['bias'] -= alpha * layer.grads['bias']


      if step % FLAGS.eval_freq == 0:
        print('Iteration ', step)

        x_ax.append(step)


        predictions = mlp.forward(X_train)
        acc_train.append(accuracy(predictions, Y_train))
        loss_tr = loss.forward(predictions, Y_train)
        loss_print_tr.append(loss_tr)

        predictions = mlp.forward(X_test)
        acc_test.append(accuracy(predictions, Y_test))
        loss_te = loss.forward(predictions, Y_test)
        loss_print_te.append(loss_te)
        print('Max train accuracy ', max(acc_train))
        print('Max test accuracy ', max(acc_test))

        print('Min train loss ', max(loss_print_tr))
        print('Min test loss ', max(loss_print_te))

  x_ax = np.array(x_ax)
  acc_test = np.array(acc_test)
  acc_train= np.array(acc_train)
  loss_print_tr = np.array(loss_print_tr)
  loss_print_te = np.array(loss_print_te)

  print('Max train accuracy ', max(acc_train))
  print('Max test accuracy ', max(acc_test))
  print('Min train loss ', min(loss_print_tr))
  print('Min test loss ', min(loss_print_te))

  fig = plt.figure()
  ax = plt.axes()

  plt.title("MLP Numpy. Accuracy curves")
  ax.plot(x_ax, acc_train, label='train');
  ax.plot(x_ax, acc_test, label='test');
  ax.set_xlabel('Step');
  ax.set_ylabel('Accuracy');
  plt.legend();
  plt.savefig('accuracy_np.jpg')

  fig = plt.figure()
  ax = plt.axes()
  plt.title("MLP Numpy. Loss curves")
  ax.plot(x_ax, loss_print_tr, label='train');
  ax.plot(x_ax, loss_print_te, label='test');
  ax.set_xlabel('Step');
  ax.set_ylabel('Loss');
  plt.legend();
  plt.savefig('loss_np.jpg')

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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
