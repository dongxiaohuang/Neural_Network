import numpy as np

def linear_forward(X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    The input X has shape (N, d_1, ..., d_K) and contains N samples with each
    example X[i] has shape (d_1, ..., d_K). Each input is reshaped to a
    vector of dimension D = d_1 * ... * d_K and then transformed to an output
    vector of dimension M.

    Args:
    - X: NumPy array of shape (N, d_1, ..., d_K) incoming data
    - W: NumPy array of shape (D, M) of weights, with D = d_1 * ... * d_K
    - b: NumPy array of shape (M, ) of biases

    Returns: Linear transformation to the incoming data
    """

    N = X.shape[0]
    D = np.prod(X.shape[1:])
    return np.dot(X.reshape(N, D), W) + b


def linear_backward(dout, X, W, b):
    """
    Computes the backward pass for a linear (fully-connected) layer.

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - X: NumPy array of shape (N, d_1, ..., d_K) incoming data
    - W: NumPy array of shape (D, M) of weights, with D = d_1 * ... * d_K
    - b: NumPy array of shape (M, ) of biases

    Returns (as tuple):
    - dX: NumPy array of shape (N, d_1, ..., d_K), gradient with respect to X
    - dW: NumPy array of shape (D, M), gradient with respect to W
    - db: NumPy array of shape (M,), gradient with respect to b
    """

    N = X.shape[0]
    D = W.shape[0]
    dX = np.dot(dout, W.T).reshape(X.shape)
    dW = np.dot(X.reshape(N, D).T, dout)
    db = dout.sum(axis = 0)
    return dX, dW, db


def relu_forward(X):
    """
    Computes the forward pass for rectified linear unit (ReLU) layer.

    Args:
    - X: NumPy array of any shape
    """

    out = X.copy()     # Must use copy to avoid pass by reference
    out[out < 0] = 0
    return out


def relu_backward(dout, X):
    """
    Computes the backward pass for rectified linear unit (ReLU) layer.

    Args:
    - dout: Upstream derivative, a NumPy array of any shape
    - X: NumPy array with the same shape as dout

    Returns: Derivative with respect to X
    """

    out = X.copy()     # Must use copy to avoid pass by reference
    out[out < 0] = 0
    out[out > 0] = 1
    return out * dout


def dropout_forward (X, p = 0.5, train = True, seed = 42):
    """
    Compute f
    Args:
    - X: Input data, a NumPy array of any shape
    - p: Dropout parameter. We drop each neuron output with probability p.
      Default is p = 0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train = True.

    Returns (as a tuple):
    - out: Output of dropout applied to X, same shape as X.
    - mask: In training mode, mask is the dropout mask
      that was used to multiply the input; in test mode, mask is None.
    """

    if seed: np.random.seed(seed)

    if train:
        mask = np.random.binomial(1, 1 - p, X.shape) / (1 - p)
        out = X * mask
    else:
        mask = None
        out = X

    return out, mask


def dropout_backward (dout, mask, p = 0.5, train = True):
    """
    Compute the backward pass for dropout
    Args:
    - dout: Upstream derivative, a numpy array of any shape
    - mask: In training mode, mask is the dropout mask that was used to
      multiply the input; in test mode, mask is None.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns:
    - dX: A numpy array, derivative with respect to X
    """

    if train: dX = dout * mask
    else:     dX = dout

    return dX
