import numpy as np
import matplotlib.pyplot as plt
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data
import pickle

model = FullyConnectedNet([512, 512, 512], input_dim=48*48*1, num_classes=7,
                          dropout=0, dtype=np.float32, reg = 0.1)
#f = open('model.pickle', 'rb')
#model = pickle.load(f)
#f.close()

data = get_FER2013_data(num_test=3589)
solver = Solver(model, data,
            update_rule='sgd_momentum',
            optim_config={
                'learning_rate': 5e-3,
            },
            lr_decay=0.95,
            num_epochs=35, batch_size=100,
            print_every=200)
solver.train()

save = input("Save model?  ")
if(save is 'y'):
    f = open('model.pickle', 'wb')
    pickle.dump(model, f)
    f.close()
    print("Model saved!")

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
"""
acc, pre = solver.check_accuracy(data['X_test'], data['y_test'])
print("acc: ", acc)

n = np.unique(data['y_test']).shape[0]
matrix = np.zeros((n,n), int)
for i in range(data['y_test'].shape[0]):
    matrix[pre[i]][data['y_test'][i]] += 1
print(matrix.T)
"""
