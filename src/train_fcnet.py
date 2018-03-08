import numpy as np
import matplotlib.pyplot as plt
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                        38   BEGIN OF YOUR CODE                            #
###########################################################################
# [20,8], reg = 0.5
model = FullyConnectedNet([50,8], reg = 0.3)
solver = Solver(model, get_CIFAR10_data(),
                update_rule='sgd',
                optim_config={
                  'learning_rate': 5e-3,
                },
                lr_decay=0.95,
                num_epochs=20, batch_size=100,
                print_every=100)
solver.train()

plt.subplot(2,1,1)
plt.title("Training loss")
plt.plot(solver.loss_history, "o")
plt.xlabel("Iteratition")

plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, "-o", label = 'train')
plt.plot(solver.val_acc_history, "-o", label = 'val')
plt.plot([0.5]* len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
plt.savefig('train_fcnet.png')

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
