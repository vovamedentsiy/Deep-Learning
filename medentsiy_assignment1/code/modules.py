"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    W = np.random.randn(out_features, in_features)*0.0001
    b =  np.zeros((out_features, 1))
    dW = np.zeros((out_features, in_features))
    db = np.zeros((out_features, 1))

    self.params = {'weight': W, 'bias': b}
    self.grads = {'weight': dW, 'bias': db}

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    W = self.params['weight']
    b = self.params['bias']
    self.last_x = x
    out = (np.dot(W, x.T) + b).T

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    x = self.last_x

    #i, b, o = x.shape[1], x.shape[0], dout.shape[1]
    dW = np.einsum('bi,bo->bio', x, dout)
    dW = (dW.sum(axis=0)).T

    db = self.grads['bias']
    db = np.dot(dout, np.eye(db.shape[0])).sum(axis=0, keepdims=True).T

    dx = np.dot(dout, self.params['weight'])

    self.grads['weight'] = dW
    self.grads['bias'] = db

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    self.last_x = x
    out = np.maximum(0, x)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    dx = dout
    dx[self.last_x <= 0] = 0

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    self.last_x = x
    m = np.max(x, axis = 1, keepdims = True)
    out = np.exp(x - m)
    out = out/out.sum(axis = 1, keepdims = True)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    M = self.forward(self.last_x)
    diag = np.eye(M.shape[1]) * M[:, np.newaxis, :]
    forw = self.forward(self.last_x)
    mi = np.einsum('bi,bo->bio', forw, forw)


    dx = diag - mi
    dx = np.einsum('ij,ijk->ik', dout, dx)

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = (-np.multiply(y, np.log(x)))
    out = out.sum(axis=1)
    out = out.mean()

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    dx = -np.divide(y, x) / x.shape[0]
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
