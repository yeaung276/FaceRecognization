# from keras import backend as K
# import tensorflow.compat.v1 as tf
# from inception_blocks_v2 import InceptionModel
# from fr_utils import load_weights_from_FaceNet
# from inception_blocks_v2_96x96 import faceRecoModel
#
# K.set_image_data_format('channels_last')
#
# model = faceRecoModel((3,96,96))
# model.summary()
# # model = InceptionModel(input_shape=(3,96,96))
# # model.summary()
# # load_weights_from_FaceNet(model)

# import scipy.io
# d = scipy.io.loadmat('test1_data.mat')
# print(d['enc'].shape)

# from os.path import isfile
# print(isfile('database.mat'))

# import scipy.io
# import numpy as np
# d = scipy.io.loadmat('data_base.mat')
# print(np.linalg.norm(d['andrew']-d['arnaud']))
# import keras
# X = keras.Input((6,6,3))
# X = keras.layers.Conv2D(32,2)(X)
# print(X.shape.as_list()-X.shape.as_list())
# import numpy as np
# from Utils import load_weights,LAYERS
# mat = np.load('net_weights.npy', allow_pickle=True).item()
# mat2 = load_weights()
#
# for layer in LAYERS:
#     print(type(mat[layer]))
#     print(type(mat2[layer]))
#     assert (mat[layer].shape == mat2[layer].shape).all(), layer
import numpy as np
# print(np.split(np.random.permutation(np.arange(10)), [5]))
# li = [1,2,3,4,5,6,7,8]
# print(li[[1,3,5,7]])
