# codes reference from VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py and
# RayXie29 /Keras-famous_CNN: https://github.com/RayXie29/Keras-famous_CNN


import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import Flatten, Input, concatenate
from keras.layers.core import Lambda
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np

K.set_image_data_format('channels_last')


class CONFIG:
    """
    input_shape: input shape of dataset
    output_units: output result dimension
    init_kernel: The kernel size for first convolution layer
    init_strides: The strides for first convolution layer
    init_filters: The filter number for first convolution layer
    regularizer: regularizer for all the convolution layers in whole NN
    initializer: weight/parameters initializer for all convolution & fc layers in whole NN
    init_maxpooling: Do the maxpooling after first two convolution layers or not
    """
    input_shape = (96, 96, 3)
    output_units = 1000,
    init_kernel = (7, 7)
    init_strides = (2, 2)
    init_filters = 64
    regularizer = l2(1e-4)
    initializer = 'he_normal'
    init_maxpooling = True


class InceptionModuleBuilder:

    def __init__(self):
        assert len(CONFIG.input_shape) == 3, "input shape should be dim 3 ( row, col, channel or channel row col )"

        self.input_shape = CONFIG.input_shape
        self.output_units = CONFIG.output_units
        self.init_kernel = CONFIG.init_kernel
        self.init_strides = CONFIG.init_strides
        self.init_filters = CONFIG.init_filters
        self.regularizer = CONFIG.regularizer
        self.initializer = CONFIG.initializer
        self.init_maxpooling = CONFIG.init_maxpooling

        if K.image_data_format() == "channels_last":

            self.row_axis = 1
            self.col_axis = 2
            self.channel_axis = 3

        else:

            self.row_axis = 2
            self.col_axis = 3
            self.channel_axis = 1

    def _cn_bn_relu(self, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                    name='inception_3a_3x3', sub_layer_index='1'):
        """
        convenient function to build convolution -> batch normalization -> relu activation layers
        """

        def f(input_x):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                       name=name + '_conv' + sub_layer_index)(input_x)
            x = BatchNormalization(axis=self.channel_axis, epsilon=0.00001, name=name + '_bn' + sub_layer_index)(x)
            x = Activation("relu")(x)

            return x

        return f

    def _inception_block(self, _1x1=64, _3x3r=96, _3x3=128, _5x5r=16, _5x5=32,
                         _pool_conv=32, strides=(1, 1), pooling="avg", pool_stride=(1, 1), name="inception3a"):
        """
        A function for building inception block, including 1x1 convolution layer, 3x3 convolution layer with dimension reducing,
        double 3x3 convolution layers with dimension reducing and maxpooling layer with dimension reducing
        :param _1x1: filter number of 1x1 convolution layer
        :param _3x3r: filter number of dimension reducing layer for 3x3 convolution layer
        :param _3x3: filter number of 3x3 convolution layer
        :param _5x5r: filter number of dimension reducing layer for double 3x3 convolution layers
        :param _5x5: filter number of double 3x3 convolution layers
        :param _maxpool: filter number of dimension reducing layer for maxpooling layer
        :return: A concatenate block of several scale convolution which is inception block
        """

        def f(input_x):
            layers = []

            # 3x3 inception conv
            branch3x3 = self._cn_bn_relu(filters=_3x3r, kernel_size=(1, 1), strides=(1, 1),
                                         padding="same", name=name + '_3x3', sub_layer_index='1')(input_x)
            branch3x3 = self._cn_bn_relu(filters=_3x3, kernel_size=(3, 3), strides=strides,
                                         padding="same", name=name + '_3x3', sub_layer_index='2')(branch3x3)
            layers.append(branch3x3)

            # 5x5 inception conv
            if _5x5 > 0:
                branch5x5 = self._cn_bn_relu(filters=_5x5r, kernel_size=(1, 1), strides=(1, 1),
                                             padding="same", name=name + '_5x5', sub_layer_index='1')(input_x)
                branch5x5 = self._cn_bn_relu(filters=_5x5, kernel_size=(5, 5), strides=strides,
                                             padding="same", name=name + '_5x5', sub_layer_index='2')(branch5x5)
                layers.append(branch5x5)

            # pooling and 1x1 conv
            if pooling == "avg":
                branchpool = AveragePooling2D(pool_size=(3, 3), strides=pool_stride)(input_x)
            elif pooling == "max":
                branchpool = MaxPooling2D(pool_size=(3, 3), strides=pool_stride)(input_x)
            if _pool_conv > 0:
                branchpool = self._cn_bn_relu(filters=_pool_conv, kernel_size=(1, 1), strides=(1, 1),
                                              padding="same", name=name + '_pool', sub_layer_index='')(branchpool)
            diff_shape = np.array(branch3x3.shape.as_list()[1::]) - np.array(branchpool.shape.as_list()[1::])
            diff_shape = np.abs(diff_shape)
            v_left = int(diff_shape[0]/2)
            v_right = diff_shape[0] - v_left
            h_left = int(diff_shape[1]/2)
            h_right = diff_shape[1] - h_left
            branchpool = ZeroPadding2D(padding=((v_left, v_right), (h_left, h_right)))(branchpool)
            layers.append(branchpool)

            # 1x1 inception conv
            if _1x1 > 0:
                branch1x1 = self._cn_bn_relu(filters=_1x1, kernel_size=(1, 1), strides=(1, 1),
                                             padding="same", name=name + '_1x1', sub_layer_index='')(input_x)
                layers.append(branch1x1)

            return concatenate(layers, axis=self.channel_axis, name=name)

        return f

    def BuildInception(self):
        """
        Main function for building inceptionV2 nn
        :return: An inceptionV2 nn
        """
        # Define the input as a tensor with shape input_shape
        X_input = Input(self.input_shape)

        # First Block
        X = Conv2D(self.init_filters, self.init_kernel,
                   strides=self.init_strides, name='conv1', padding='same')(X_input)
        X = BatchNormalization(axis=self.channel_axis, name='bn1')(X)
        X = Activation('relu')(X)

        # MAXPOOL
        if self.init_maxpooling:
            X = MaxPooling2D((3, 3), strides=2, padding='same')(X)

        # Second Block
        X = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv2')(X)
        X = BatchNormalization(axis=self.channel_axis, epsilon=0.00001, name='bn2')(X)
        X = Activation('relu')(X)

        # Third Block
        X = Conv2D(192, (3, 3), strides=(1, 1), padding='same', name='conv3')(X)
        X = BatchNormalization(axis=self.channel_axis, epsilon=0.00001, name='bn3')(X)
        X = Activation('relu')(X)

        # MAXPOOL
        X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

        # inception block: 1(a/b/c)
        X = self._inception_block(_1x1=64, _3x3r=96, _3x3=128, _5x5r=16, _5x5=32,
                                  _pool_conv=32, pooling='max', pool_stride=(2, 2), name='inception_3a')(X)
        X = self._inception_block(_1x1=64, _3x3r=96, _3x3=128, _5x5r=32, _5x5=64,
                                  _pool_conv=64, pooling='avg', pool_stride=(3, 3), name='inception_3b')(X)
        X = self._inception_block(_1x1=0, _3x3r=128, _3x3=256, _5x5r=32, _5x5=64, _pool_conv=0,
                                  pooling='max', strides=(2, 2), pool_stride=(2, 2), name='inception_3c')(X)

        # inception block: 2(a/e)
        X = self._inception_block(_1x1=256, _3x3r=96, _3x3=192, _5x5r=32, _5x5=64,
                                  _pool_conv=128, pooling='avg', pool_stride=(3, 3), name='inception_4a')(X)
        X = self._inception_block(_1x1=0, _3x3r=160, _3x3=256, _5x5r=64, _5x5=128, _pool_conv=0,
                                  pooling='max', strides=(2, 2), pool_stride=(2, 2), name='inception_4e')(X)

        # inception block: 3(a/b)
        X = self._inception_block(_1x1=256, _3x3r=96, _3x3=384, _5x5r=0, _5x5=0,
                                  _pool_conv=96, pooling='avg', pool_stride=(3, 3), name='inception_5a')(X)
        X = self._inception_block(_1x1=256, _3x3r=96, _3x3=384, _5x5r=0, _5x5=0,
                                  _pool_conv=96, pooling='max', pool_stride=(2, 2), name='inception_5b')(X)

        # Top Layer
        X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(X)
        X = Flatten()(X)
        X = Dense(128, name='dense_layer')(X)

        # L2 normalization
        X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

        # Create Model Instance
        model = Model(inputs=X_input, outputs=X, name='FaceNet')

        return model


##################################################################################################

# inception builder 2
# noinspection PyPep8Naming
def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, name=layer + '_conv' + num)(
        x)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer + '_bn' + num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding)(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, name=layer + '_conv' + '2')(
        tensor)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer + '_bn' + '2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor


def inception_block_1a(X):
    """
    Implementation of an inception block
    """

    X_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1))(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)

    X_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2))(X_5x5)
    X_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = MaxPooling2D(pool_size=3, strides=2)(X)
    X_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(X_pool)

    X_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)

    # CONCAT
    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=3)

    return inception


def inception_block_1b(X):
    X_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1))(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)

    X_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2))(X_5x5)
    X_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(X)
    X_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=(4, 4))(X_pool)

    X_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)

    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=3)

    return inception


def inception_block_1c(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_3c_3x3',
                      cv1_out=128,
                      cv1_filter=(1, 1),
                      cv2_out=256,
                      cv2_filter=(3, 3),
                      cv2_strides=(2, 2),
                      padding=(1, 1))

    X_5x5 = conv2d_bn(X,
                      layer='inception_3c_5x5',
                      cv1_out=32,
                      cv1_filter=(1, 1),
                      cv2_out=64,
                      cv2_filter=(5, 5),
                      cv2_strides=(2, 2),
                      padding=(2, 2))

    X_pool = MaxPooling2D(pool_size=3, strides=2)(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(X_pool)

    inception = concatenate([X_3x3, X_5x5, X_pool], axis=3)

    return inception


def inception_block_2a(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_4a_3x3',
                      cv1_out=96,
                      cv1_filter=(1, 1),
                      cv2_out=192,
                      cv2_filter=(3, 3),
                      cv2_strides=(1, 1),
                      padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                      layer='inception_4a_5x5',
                      cv1_out=32,
                      cv1_filter=(1, 1),
                      cv2_out=64,
                      cv2_filter=(5, 5),
                      cv2_strides=(1, 1),
                      padding=(2, 2))

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(X)
    X_pool = conv2d_bn(X_pool,
                       layer='inception_4a_pool',
                       cv1_out=128,
                       cv1_filter=(1, 1),
                       padding=(2, 2))
    X_1x1 = conv2d_bn(X,
                      layer='inception_4a_1x1',
                      cv1_out=256,
                      cv1_filter=(1, 1))
    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=3)

    return inception


def inception_block_2b(X):
    # inception4e
    X_3x3 = conv2d_bn(X,
                      layer='inception_4e_3x3',
                      cv1_out=160,
                      cv1_filter=(1, 1),
                      cv2_out=256,
                      cv2_filter=(3, 3),
                      cv2_strides=(2, 2),
                      padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                      layer='inception_4e_5x5',
                      cv1_out=64,
                      cv1_filter=(1, 1),
                      cv2_out=128,
                      cv2_filter=(5, 5),
                      cv2_strides=(2, 2),
                      padding=(2, 2))

    X_pool = MaxPooling2D(pool_size=3, strides=2)(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(X_pool)

    inception = concatenate([X_3x3, X_5x5, X_pool], axis=3)

    return inception


def inception_block_3a(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_5a_3x3',
                      cv1_out=96,
                      cv1_filter=(1, 1),
                      cv2_out=384,
                      cv2_filter=(3, 3),
                      cv2_strides=(1, 1),
                      padding=(1, 1))
    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(X)
    X_pool = conv2d_bn(X_pool,
                       layer='inception_5a_pool',
                       cv1_out=96,
                       cv1_filter=(1, 1),
                       padding=(1, 1))
    X_1x1 = conv2d_bn(X,
                      layer='inception_5a_1x1',
                      cv1_out=256,
                      cv1_filter=(1, 1))

    inception = concatenate([X_3x3, X_pool, X_1x1], axis=3)

    return inception


def inception_block_3b(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_5b_3x3',
                      cv1_out=96,
                      cv1_filter=(1, 1),
                      cv2_out=384,
                      cv2_filter=(3, 3),
                      cv2_strides=(1, 1),
                      padding=(1, 1))
    X_pool = MaxPooling2D(pool_size=3, strides=2)(X)
    X_pool = conv2d_bn(X_pool,
                       layer='inception_5b_pool',
                       cv1_out=96,
                       cv1_filter=(1, 1))
    X_pool = ZeroPadding2D(padding=(1, 1))(X_pool)

    X_1x1 = conv2d_bn(X,
                      layer='inception_5b_1x1',
                      cv1_out=256,
                      cv1_filter=(1, 1))
    inception = concatenate([X_3x3, X_pool, X_1x1], axis=3)

    return inception


def InceptionModel(input_shape):
    """
    Implementation of the Inception model used for FaceNet

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # First Block
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=2)(X)

    # Second Block
    X = Conv2D(64, (1, 1), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)

    # Second Block
    X = Conv2D(192, (3, 3), strides=(1, 1), name='conv3')(X)
    X = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size=3, strides=2)(X)

    # Inception 1: a/b/c
    X = inception_block_1a(X)
    X = inception_block_1b(X)
    X = inception_block_1c(X)

    # Inception 2: a/b
    X = inception_block_2a(X)
    X = inception_block_2b(X)

    # Inception 3: a/b
    X = inception_block_3a(X)
    X = inception_block_3b(X)

    # Top layer
    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(X)
    X = Flatten()(X)
    X = Dense(128, name='dense_layer')(X)

    # L2 normalization
    X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

    # Create model instance
    model = Model(inputs=X_input, outputs=X, name='FaceRecoModel')

    return model

    # TODO: check out this repo: https://github.com/shubham0204/Face_Recognition_with_TF
