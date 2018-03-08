from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import pandas as pd
from scipy.misc import imread
import platform
import pickle



def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def get_FER2013_data(num_training=27000, num_validation=1000, num_test=3500,
                     subtract_mean=True):
    """
    Load the FER2013 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    with open('src/utils/data.pickle', 'rb') as handle:
        data = pickle.load(handle)
    X_train = data['X_train'][0:num_training].astype(np.float32)
    y_train = data['y_train'][0:num_training]
    X_val = data['X_train'][num_training:num_training + num_validation].astype(np.float32)
    y_val = data['y_train'][num_training:num_training + num_validation]
    X_test = data['X_test'][0:num_test].astype(np.float32)
    y_test = data['y_test'][0:num_test]

    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        f = open('mean.pickle', 'wb')
        pickle.dump(mean_image, f)
        f.close()
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
            'X_train': X_train, 'y_train': np.array(y_train),
            'X_val': X_val, 'y_val': np.array(y_val),
            'X_test': X_test, 'y_test': np.array(y_test),
    }
def get_FER2013_Train_Test_data():
    # Load the raw CIFAR-10 data
    fer_dir = '/vol/bitbucket/395ML_NN_Data/datasets/FER2013'
    fer_train_dir = '/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Train'
    fer_test_dir = '/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test'

    # read file as dictionary
    label_dic = {}
    label_file_dir = '/vol/bitbucket/395ML_NN_Data/datasets/FER2013/labels_public.txt'
    with open(label_file_dir) as f:
        next(f) # skip first line
        for line in f:
            (tag, label) = line.split(',')
            label_dic[tag] = int(label[0])
    with open('label_dic.pickle', 'wb') as handle:
        pickle.dump(label_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('label_dic.pickle', 'rb') as handle:
    #     label_dic1 = pickle.load(handle)
    X_train = []
    y_train = []
    y_test = []
    for f in os.listdir(fer_train_dir):
        fig_dir = fer_train_dir +'/'+ f
        fig = imread(name = fig_dir, mode = 'L').reshape(48,48,1)
        X_train.append(fig)
        y_train.append(label_dic["Train/"+str(f)])
        if(str(f) == '1.jpg'):
            print(0 == label_dic["Train/"+str(f)])
        if(str(f) == '15732.jpg'):
            print(4 == label_dic["Train/"+str(f)])
        if(str(f) == '15800.jpg'):
            print(3 == label_dic["Train/"+str(f)])

    X_test = []
    for f in os.listdir(fer_test_dir):
        fig_dir = fer_test_dir +'/'+ f
        fig = imread(name = fig_dir, mode = 'L').reshape(48,48,1)
        X_test.append(fig)
        y_test.append(label_dic["Test/"+str(f)])
        if(str(f) == '32290.jpg'):
            print(1 == label_dic["Test/"+str(f)])
        if(str(f) == '32285.jpg'):
            print(5 == label_dic["Test/"+str(f)])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    data = {'X_train': X_train, 'y_train': y_train,
    'X_test': X_test, 'y_test': y_test}
    with open('data.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('data.pickle', 'rb') as handle:
    #     data = pickle.load(handle)

    return data
