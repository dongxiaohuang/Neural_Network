import numpy as np

def softmax (logits, y):
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

    X = logits.max(axis = 1, keepdims = True)        # For numerical stability
    sigma = np.exp(logits - X)                       # Unnormalized probabilities
    sigma /= sigma.sum(axis = 1, keepdims = True)    # Each row sums to 1

    N = y.shape[0]
    loss = np.log(sigma)
    loss = - loss[range(N), y].mean()

    dlogits = sigma.copy()
    dlogits[range(N), y] -= 1
    dlogits /= N
    return loss, dlogits
