import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data
import pickle

def test_fer_model(img_folder, model="/src/utils/model.pickle"):
    """
    Given a folder with images, load the images and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ### Start your code here

    #X, y = load_image(img_folder)

    f = open(model, 'rb')
    fcn_model = pickle.load(f)
    f.close()

    #preds = solver.check_accuracy(X, y)

    ### End of code
    return preds

#model = FullyConnectedNet([128, 32], input_dim=48*48*1, num_classes=7,\
#                          dropout=0, seed=42, reg = 0.3)
f = open('model.pickle', 'rb')
model = pickle.load(f)
f.close()

data = get_FER2013_data()
solver = Solver(model, data,
            update_rule='sgd_momentum',
            optim_config={
                'learning_rate': 1e-3,
            },
            lr_decay=0.95,
            num_epochs=100, batch_size=100,
            print_every=200)
#solver.train()

#f = open('model.pickle', 'wb')
#pickle.dump(model, f)
#f.close()

print("acc: ", solver.check_accuracy(data['X_test'], data['y_test']))
