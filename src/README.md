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
model is saved in utils/bestmodels/weights.149-1.11.hdf5
`from test_cnn import test_deep_fer_model`
Args:
  - img_folder: Path to the images to be tested
  - model: Path to the model to be loaded
  Returns:
  - preds: A numpy vector of size N with N being the number of images in
  img_folder.
