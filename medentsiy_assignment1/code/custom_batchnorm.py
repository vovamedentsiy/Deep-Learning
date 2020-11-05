import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object.

    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    self.gamma = torch.nn.Parameter(torch.ones(n_neurons))
    self.beta = torch.nn.Parameter(torch.zeros(n_neurons))
    self.n_neurons = n_neurons
    self.eps = eps

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization

    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    if input.size()[1] == self.n_neurons:
        mean = input.mean(dim = 0)
        std = input.std(dim = 0)
        x_tilda = input-mean[None, :]
        stable_std = torch.sqrt(std*std + self.eps)
        x_tilda  = x_tilda/stable_std
        out = x_tilda
    else:
        print('Incorrect shape')
    ########################
    # END OF YOUR CODE    #
    #######################

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization

    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    ctx.save_for_backward(input, gamma, beta)

    mean = input.mean(dim = 0)
    var = input.var(dim = 0, unbiased=False)
    x_tilda = input-mean[None, :]
    stable_std = torch.sqrt(var + eps)

    ctx.constant_std = stable_std
    ctx.constant_mu = mean

    x_tilda  = x_tilda/stable_std

    ctx.constant_tilda = x_tilda

    y = x_tilda * gamma[None, :] + beta[None, :]
    out = y

    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.

    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments

    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x_tilda = ctx.constant_tilda
    mean = ctx.constant_mu
    stable_std = ctx.constant_std


    x, gamma, beta = ctx.saved_variables
    grad_input = grad_gamma = grad_beta = None

    B = x.size()[0]

    if ctx.needs_input_grad[2]:
      grad_beta = grad_output.sum(dim = 0)

    if ctx.needs_input_grad[1]:
      grad_gamma = grad_output*x_tilda
      grad_gamma = grad_gamma.sum(dim = 0)

    if ctx.needs_input_grad[0]:
      grad_x_tilda  =  grad_output * gamma[None, :]
      grad_std = (- 1/2 * grad_x_tilda * (x - mean[None, :])*(stable_std[None, :]).pow(-3)).sum(dim = 0)
      grad_mu = (- grad_x_tilda / stable_std).sum(dim = 0) +  grad_std * (-2*(x-mean[None, :]).sum(dim = 0)/B)
      grad_input =  grad_x_tilda / stable_std + grad_std[None, :] * 2 * (x - mean[None, :])/B + grad_mu[None, :] / B
    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.

    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_neurons = n_neurons
    self.eps  = eps

    self.gamma = torch.nn.Parameter(torch.ones(n_neurons))
    self.beta = torch.nn.Parameter(torch.zeros(n_neurons))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction

    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    if input.size()[1] == self.n_neurons:
        bn_manual = CustomBatchNormManualFunction(self.n_neurons)
        out = bn_manual.apply(input, self.gamma, self.beta, self.eps)
    else:
        print('Incorrect shape')
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
