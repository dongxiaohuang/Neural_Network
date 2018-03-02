import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data

def test_fer_model(img_folder, model="/path/to/model"):
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

    ### End of code
    return preds

model = FullyConnectedNet([512,128], input_dim=48*48*1, num_classes=7, reg = 0.0)
solver = Solver(model, get_FER2013_data(),
            update_rule='sgd_momentum',
            optim_config={
                'learning_rate': 1e-3,
            },
            lr_decay=0.95,
            num_epochs=10, batch_size=100,
            print_every=100)
solver.train()

