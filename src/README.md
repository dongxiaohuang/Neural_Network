# To Test Model for Q5
Files <i>test.py</i>, <i>model.pickle</i>, and <i>mean.pickle</i> are required in the working directory. The model is stored in file <i>model.pickle</i>.

To test the model, import <i>test.py</i> as:
```
import numpy
from test import test_fer_model
preds = test_fer_model(img_folder, 'model.pickle')
```
Args:
	- img_folder: path to the test images
	- model: path to the model

Return:
	- preds: A numpy vector of size N with N being the number of images in img_folder

As the data normalization is used in training stage, the mean of training images must be loaded in order to get correct predictions. The mean of training images is stored in the file <i>mean.pickle</i>, put the file in the same directory where you run your program.

# To Test Model for Q6
Install python models and run in python 3.
## Prerequisite
- install pillow:
`pip install Pillow`
- Before installing Keras, install one of its backend engines: TensorFlow:
`pip install tensorflow`
- install keras:
`pip install keras`
- HDF5 and h5py (required to load saved Keras models from disk).
`pip install h5py`

## Test for Q6
The model is saved in weights.149-1.11.hdf5.

`from test_cnn import test_deep_fer_model`

Args:
  - img_folder: Path to the images to be tested
  - model: Path to the model to be loaded

Return:
  - preds: A numpy vector of size N with N being the number of images in img_folder

The data normalization is also used in the training period, we include the pickle file <i>mean_image.pickle </i>. So when using the function `test_deep_fer_model`, the file should be read. In order to ensure the pickle is read functionally, the pickle file should be placed in the same directory of the file <i>test_cnn.py</i>
and then in the same directory to run the code.

for example: if we are in the directory <i>utils</i>, and the file <i>test_cnn, mean_image.pickle, bestmodels/weights.149-1.11.hdf5</i> are all inside the directory. and the images path is :'/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test', then in this directory <i>utils</i>, type the following code.
```
python
from test_cnn import test_deep_fer_model
y_predict = test_deep_fer_model('/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test', './bestmodels/weights.149-1.11.hdf5')
```
