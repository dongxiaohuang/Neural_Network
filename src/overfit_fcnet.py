import numpy as np
import matplotlib.pyplot as plt
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

plt.subplot(2,1,1)
plt.title("Training loss")
plt.plot(solver.loss_history, "o")
plt.xlabel("Iteratition")

plt.subplot(2,1,2)
plt.title('Overfit Network Accuracy')
plt.plot(solver.train_acc_history, "-o", label = 'train')
plt.plot(solver.val_acc_history, "-o", label = 'val')
plt.plot([0.5]* len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)

plt.savefig('overfit_fcnet.png')
plt.show()
