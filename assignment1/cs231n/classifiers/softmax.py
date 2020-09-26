from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_trains = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_trains):
      scores = X[i].dot(W)
      scores -= np.max(scores) 
      scores = np.exp(scores)
      scores /= np.sum(scores)
      p = scores[y[i]]
      loss -= np.log(p)
      for j in range(num_classes):
        if j == y[i]:
          dW.T[j] += (p - 1)*X[i]
        else:
          dW.T[j] += scores[j]*X[i]
    
    loss /= num_trains
    loss += reg*np.sum(W*W)
    dW /= num_trains
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_trains = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores)
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1)[:,np.newaxis]
    loss = np.mean(-np.log(scores[np.arange(y.shape[0]),y]))
    loss += reg*np.sum(W*W)
    results = np.zeros_like(scores)
    results[np.arange(y.shape[0]),y] = 1
    dW = X.T.dot(scores - results) 
    dW /= num_trains
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
