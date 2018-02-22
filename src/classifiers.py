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
    loss, dlogits = None, None

    n = y.shape[0]
    X = -logits.max(axis = 1, keepdims = True)     # For numerical stability
    sigma_z = np.exp(logits + X)                          # Unnormalized probabilities
    sigma_z /= sigma_z.sum(axis = 1, keepdims = True)     # Each row sums to 1
    lg_sigma_z = np.log(sigma_z)
    loss = -lg_sigma_z[range(n), y].sum() / n
    dlogits = sigma_z.copy()
    dlogits[range(n), y] -= 1
    dlogits /= n

    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """


    return loss, dlogits
