import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

# Overfit the network with 50 samples of CIFAR-10

model = FullyConnectedNet([50], reg = 0)
data = get_CIFAR10_data(num_training = 50)
solver = Solver(model, data,
                update_rule = 'sgd',
                optim_config = {
                  'learning_rate': 1e-3,
                },
                lr_decay = 0.95,
                num_epochs = 20, batch_size = 100,
                print_every=100)
solver.train()
