import scipy.io
from Utils.Utils import plot_history
from keras.utils import plot_model
from Model.similarity import ModelBuilder


mat = scipy.io.loadmat('0.0.1_hist.mat')
fig = plot_history(mat, save='model_train_history.png')

model = ModelBuilder.buildModel()
model.load_weights('model.h5')
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


