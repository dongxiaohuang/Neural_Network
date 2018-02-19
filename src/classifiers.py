import numpy as np

def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: NumPy array of shape (N, C)
    - y: NumPy array of shape (N, ). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """

    X -= X.max(axis = 1, keepdims = True)     # For numerical stability
    prob = np.exp(X)                          # Unnormalized probabilities
    prob /= prob.sum(axis = 1, keepdims = True)     # Rows sum to 1
    loss = np.log(prob)
    loss = loss[np.arange(len(loss)), y].sum()

    dlogits = None

    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """

    return loss, dlogits
