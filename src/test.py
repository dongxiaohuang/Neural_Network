import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import listdir
from scipy.misc import imread

def test_fer_model(img_folder, model="model.pickle"):
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

    f = open(model, 'rb')
    fcn_model = pickle.load(f)
    f.close()
    f = open('mean.pickle', 'rb')
    mean = pickle.load(f)
    f.close()

    X = []
    print("loading data...")
    for f in listdir(img_folder):
        fig_dir = img_folder +'/'+ f
        fig = imread(name = fig_dir, mode = 'L').reshape(48,48,1)
        X.append(fig)
    print("Loading finished!")

    X = np.array(X).astype(np.float32)
    X -= mean
    X = X.transpose(0, 3, 1, 2).copy()

    preds = np.argmax(fcn_model.loss(X), axis=1)

    ### End of code
    return preds

print (test_fer_model('/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test'))
