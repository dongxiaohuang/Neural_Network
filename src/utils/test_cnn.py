import numpy as np
from os import listdir
from scipy.misc import imread
from keras.models import load_model

def test_deep_fer_model(img_folder, model= './utils/bestmodels/weights.23-1.20.hdf5'):
###########################################load model#####################
    """
    Given a folder with images, load the images and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """

    test_model = load_model(model)
    preds = None
    ### Start your code here
    X_test = []
    print("loading data...")
    for f in listdir(img_folder):
        fig_dir = img_folder +'/'+ f
        fig = imread(name = fig_dir, mode = 'L').reshape(48,48,1)
        X_test.append(fig)
    print("Loading finished!")
    X_test = np.array(X_test)

    with open('mean_image.pickle', 'rb') as handle:
        mean_image = pickle.load(handle)

    X_test = X_test.astype('float64')
    X_test -= mean_image
    
    X_test /= np.max(X_test) # Normalise data to [0, 1] range



    test_model = load_model(model)
    Y_predict = test_model.predict(X_test, batch_size=None, verbose=1, steps=None)
    preds = np.argmax(Y_predict, axis=1)
    ### End of code
    return preds

print (test_deep_fer_model('/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test'))
