# To Test Model for Q5
Files <i>test_fcn.py</i>, <i>model.pkl</i>, <i>mean.pkl</i>, and <i>src</i> are required in the working directory. The model is stored in file <i>model.pkl</i>.

To test the model, import <i>test_fcn.py</i> as:
```
from test_fcn import test_fer_model
preds = test_fer_model(img_folder, 'model.pkl')
```
Args:
  - img_folder: Path to the test images
  - model: Path to the model

Return:
  - preds: A numpy vector of size N with N being the number of images in img_folder

As the data normalization is used in training stage, the mean of training images must be loaded in order to get correct predictions. The mean of training images is stored in the file <i>mean.pkl</i>, put the file in the same directory where you run your program.

# To Test Model for Q6
Files <i>test_cnn.py</i>, <i>cnn_model.hdf5</i>, and <i>mean_image.pkl</i> are required in the working directory. Install python models and run in python 3.

## Dependencies
- install pillow:
`pip install Pillow`
- Before installing Keras, install one of its backend engines: TensorFlow:
`pip install tensorflow`
- install keras:
`pip install keras`
- HDF5 and h5py (required to load saved Keras models from disk).
`pip install h5py`

## Test for Q6
The model is saved in cnn_model.hdf5.

```
from test_cnn import test_deep_fer_model
preds = test_deep_fer_model(img_folder, model)
```
Args:
  - img_folder: path to the images to be tested
  - model: path to the model to be loaded

Return:
  - preds: a numpy vector of size N with N being the number of images in img_folder

The data normalization is also used in the training period, the mean of training images is stored in the pickle file <i>mean_image.pickle </i>. So when using the function `test_deep_fer_model`, the file must be in the working directory.

for example: if we are in the directory <i>utils</i>, and the file <i>test_cnn, mean_image.pkl, cnn_model.hdf5</i> are all inside the directory, the images path is :'/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test', in the directory <i>utils</i>, type the following code:

```
from test_cnn import test_deep_fer_model
y_predict = test_deep_fer_model('/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test', 'cnn_model.hdf5')
```
