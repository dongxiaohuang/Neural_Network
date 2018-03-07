import numpy as np

from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)


def random_init (n_in, n_out, weight_scale = 5e-2, dtype = np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: Number of input nodes into each output
    - n_out: Number of output nodes for each input
    """

    W = np.random.randn(n_in, n_out) * weight_scale
    b = np.zeros(n_out)
    return W.astype(dtype), b.astype(dtype)


class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:

    [linear - relu - (dropout)] x (N - 1) - linear - softmax

    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """

    def __init__(self, hidden_dims, input_dim = 32*32*3, num_classes = 10,
                 dropout = 0, reg = 0.0, weight_scale = 1e-2, dtype = np.float32,
                 seed = None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: List of the sizes of each hidden layer
        - input_dim: Integer giving the input's size
        - num_classes: Number of classes to classify
        - dropout: A scalar between 0. and 1. determining the dropout factor.
          If dropout = 0, then dropout is not applied.
        - reg: Regularisation factor
        """

        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed: np.random.seed(seed)
        self.params = dict()

        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """

        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            Wi, bi = 'W' + str(i + 1), 'b' + str(i + 1)
            self.params[Wi], self.params[bi] = \
                    random_init(dims[i], dims[i + 1], weight_scale, self.dtype)

        """
        When using dropout we need to pass a dropout_param dictionary to
        each dropout layer so that the layer knows the dropout probability
        and the mode (train / test). You can pass the same dropout_param to
        each dropout layer.
        """

        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y = None):
        """
        Compute loss and gradient for a minibatch of data.

        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i]

        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """

        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()

        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """

        if self.use_dropout:
            self.dropout_params["train"] = False if y is None else True

        Xi = linear_cache['0'] = X
        if self.use_dropout:
            p, t = self.dropout_params["p"],     \
                      self.dropout_params["train"]
        for i in range(self.num_layers):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
            Xi = relu_cache[str(i)] = linear_forward(Xi, W, b)
            if i != self.num_layers-1:
                Xi = linear_cache[str(i+1)] = relu_forward(Xi)
                if self.use_dropout:
                    # receive (out, mask)
                    Xi, dropout_cache[str(i)] = dropout_forward(Xi, \
                            p, t, None)
        scores = Xi

        # If y is None then we are in test mode, so just return scores
        if y is None: return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """

        loss, dX = softmax(scores, y)
        for i in reversed(range(self.num_layers)):
            if i != self.num_layers - 1:
                if self.use_dropout:
                    dX = dropout_backward(dX, \
                            dropout_cache[str(i)], p=p, train=t)
                dX = relu_backward(dX, relu_cache[str(i)])
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
            dX, dW, db = linear_backward(dX, linear_cache[str(i)], W, b)
            grads['W' + str(i + 1)], grads['b' + str(i + 1)] = \
                    dW + self.reg * self.params['W' + str(i + 1)], db
            loss += 0.5 * self.reg * np.sum(W**2)

        return loss, grads
