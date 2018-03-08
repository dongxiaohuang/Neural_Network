# To Test Model For Q6
please install python models and run in python 3.
## Prerequisite
- install pillow:
`pip install Pillow`
- Before installing Keras, please install one of its backend engines: TensorFlow:
`pip install tensorflow`
- install keras:
`pip install keras`
- HDF5 and h5py (required to load saved Keras models from disk).
`pip install h5py`

## Test for Q6
model is saved in utils/bestmodels/cnn_model.hdf5
`from test_cnn import test_deep_fer_model`
Args:
  - img_folder: Path to the images to be tested
  - model: Path to the model to be loaded
  Returns:
  - preds: A numpy vector of size N with N being the number of images in
  img_folder.

As the data normalization was used in the training period, so we include the pickle file <i>mean_image.pickle </i>, so when used the function `test_deep_fer_model`, the file should be read. In order to ensure the pickle is read functionally, the pickle file should be placed in the same directory of the file <i>test_cnn.py</i>
and then in the same directory to run the code.

for example: if we are in the directory <i>utils</i>, and the file <i>test_cnn, mean_image.pickle, bestmodels/cnn_model.hdf5</i> are all inside the directory. and the images path is :'/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test', then in this directory <i>utils</i>, type the following code.
```
python
from test_cnn import test_deep_fer_model
y_predict = test_deep_fer_model('/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test', './bestmodels/cnn_model.hdf5')
```
