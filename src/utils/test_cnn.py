from keras.models import Model # basic class for specifying and training a neural network
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import os
from scipy.misc import imread
from sklearn.metrics import confusion_matrix
from keras.models import load_model

def test_deep_fer_model(img_folder, model= './utils/bestmodels/weights.23-1.20.hdf5'):
###########################################load model#####################
    test_model = load_model(model)
    preds = None
    ### Start your code here
    X_test = []
    print("loading data...")
    for f in os.listdir(img_folder):
        fig_dir = img_folder +'/'+ f
        fig = imread(name = fig_dir, mode = 'L').reshape(48,48,1)
        X_test.append(fig)
    print("Loading finished!")
    X_test = np.array(X_test)

    test_model = load_model(model)
    Y_predict = test_model.predict(X_test, batch_size=None, verbose=1, steps=None)
    preds = np.argmax(Y_predict, axis=1)
    ### End of code
    return preds


# score = model_resume.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
# Y_predict = model_resume.predict(X_test, batch_size=None, verbose=1, steps=None)
# y_predict = np.argmax(Y_predict, axis=1)
#
# cm = confusion_matrix(y_test,y_predict)
# print(cm)
#rec_pre = pf.recall_precision_rates(num_classes, cm)
#f1 = pf.fa_measure(1, num_classes, rec_pre)
#cr = pf.all_classfi_rate(cm)

#print(cr)
#print(f1)

# print('loss : %.2f'%score[0])
# print('acc : %.2f'%score[1]*100)
# checkpoint
print (test_deep_fer_model('/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test'))
