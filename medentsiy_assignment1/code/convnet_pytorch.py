"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object.

    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem


    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels = n_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
    self.bn_1 = nn.BatchNorm2d(64)
    self.relu_1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)


    self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, bias = True)
    self.bn_2 = nn.BatchNorm2d(128)
    self.relu_2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    self.conv3_a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
    self.bn_3_a = nn.BatchNorm2d(256)
    self.relu_3_a = nn.ReLU()

    self.conv3_b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
    self.bn_3_b = nn.BatchNorm2d(256)
    self.relu_3_b = nn.ReLU()
    self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    self.conv4_a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
    self.bn_4_a = nn.BatchNorm2d(512)
    self.relu_4_a = nn.ReLU()
    self.conv4_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
    self.bn_4_b = nn.BatchNorm2d(512)
    self.relu_4_b = nn.ReLU()
    self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    self.conv5_a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
    self.bn_5_a = nn.BatchNorm2d(512)
    self.relu_5_a = nn.ReLU()
    self.conv5_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
    self.bn_5_b = nn.BatchNorm2d(512)
    self.relu_5_b = nn.ReLU()
    self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)

    self.linear = nn.Linear(512, n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = self.pool1(self.relu_1(self.bn_1(self.conv1(x))))
    out = self.pool2(self.relu_2(self.bn_2(self.conv2(out))))
    out = self.relu_3_a(self.bn_3_a(self.conv3_a(out)))
    out = self.pool3(self.relu_3_b(self.bn_3_b(self.conv3_b(out))))
    out = self.relu_4_a(self.bn_4_a(self.conv4_a(out)))
    out = self.pool4(self.relu_4_b(self.bn_4_b(self.conv4_b(out))))
    out = self.relu_5_a(self.bn_5_a(self.conv5_a(out)))
    out = self.pool5(self.relu_5_b(self.bn_5_b(self.conv5_b(out))))
    out = self.avgpool(out)
    out = out.view(-1, 512)
    out = self.linear(out)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out
