import numpy as np
from keras import layers, models
from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense
from keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D, concatenate
from keras.layers import Flatten, Lambda

########### Conv -> Batch Normalization -> Activation ######
class CnBnRelu(layers.Layer):
    def __init__(self, weights, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                    name='inception_3a_3x3', sub_layer_index='1', channel_axis=3):
        super().__init__(name=name)
        self.weights_ = weights
        self.sub_layer_index = sub_layer_index
        self.cn = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                    name=name + '_conv' + sub_layer_index)
        self.bn = BatchNormalization(axis=channel_axis, epsilon=0.00001, name=name + '_bn' + sub_layer_index)
        self.relu = Activation("relu")
        
    def build(self, input_shape):
        self.cn.build(input_shape)
        self.bn.build(self.cn.compute_output_shape(input_shape))
        self.cn.set_weights(self.weights_[self.name + '_conv' + self.sub_layer_index])
        self.bn.set_weights(self.weights_[self.name + '_bn' + self.sub_layer_index])
        
    def call(self, input):
        x = self.cn(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


################# Inception layer signature ################
class InceptionLayer(layers.Layer):
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
    def __init__(self, weights, _1x1=64, 
                _3x3r=96, _3x3=128, 
                _5x5r=16, _5x5=32,
                strides=(1, 1),
                _pool_conv=32, pooling="avg", pool_stride=(1, 1), 
                name="inception_3a", channel_axis=3):
        super().__init__(name=name)
        self.weights_ = weights
        self.channel_axis = channel_axis
        self.strides=strides
        self._1x1 = _1x1
        self._3x3r = _3x3r
        self._3x3 = _3x3
        self._5x5r = _5x5r
        self._5x5 = _5x5
        self._pool_conv = _pool_conv
        self.pooling = pooling
        self.pool_stride = pool_stride
        
    def call(self, input_x):
        layers = []

        # 3x3 inception conv
        branch3x3 = CnBnRelu(self.weights_,filters=self._3x3r, kernel_size=(1, 1), strides=(1, 1),
                                        padding="same", name=self.name + '_3x3', sub_layer_index='1')(input_x)
        branch3x3 = CnBnRelu(self.weights_,filters=self._3x3, kernel_size=(3, 3), strides=self.strides,
                                        padding="same", name=self.name + '_3x3', sub_layer_index='2')(branch3x3)
        layers.append(branch3x3)

        # 5x5 inception conv
        if self._5x5 > 0:
            branch5x5 = CnBnRelu(self.weights_,filters=self._5x5r, kernel_size=(1, 1), strides=(1, 1),
                                            padding="same", name=self.name + '_5x5', sub_layer_index='1')(input_x)
            branch5x5 = CnBnRelu(self.weights_,filters=self._5x5, kernel_size=(5, 5), strides=self.strides,
                                            padding="same", name=self.name + '_5x5', sub_layer_index='2')(branch5x5)
            layers.append(branch5x5)

        # pooling and 1x1 conv
        if self.pooling == "avg":
            branchpool = AveragePooling2D(pool_size=(3, 3), strides=self.pool_stride)(input_x)
        elif self.pooling == "max":
            branchpool = MaxPooling2D(pool_size=(3, 3), strides=self.pool_stride)(input_x)
        if self._pool_conv > 0:
            branchpool = CnBnRelu(self.weights_,filters=self._pool_conv, kernel_size=(1, 1), strides=(1, 1),
                                            padding="same", name=self.name + '_pool', sub_layer_index='')(branchpool)
        diff_shape = np.array(branch3x3.shape.as_list()[1::]) - np.array(branchpool.shape.as_list()[1::])
        diff_shape = np.abs(diff_shape)
        v_left = int(diff_shape[0]/2)
        v_right = diff_shape[0] - v_left
        h_left = int(diff_shape[1]/2)
        h_right = diff_shape[1] - h_left
        branchpool = ZeroPadding2D(padding=((v_left, v_right), (h_left, h_right)))(branchpool)
        layers.append(branchpool)

        # 1x1 inception conv
        if self._1x1 > 0:
            branch1x1 = CnBnRelu(self.weights_, filters=self._1x1, kernel_size=(1, 1), strides=(1, 1),
                                            padding="same", name=self.name + '_1x1', sub_layer_index='')(input_x)
            layers.append(branch1x1)

        return concatenate(layers, axis=self.channel_axis, name=self.name)


##################### Starting layer #######################
class StartingLayer(layers.Layer):
    def __init__(self, weights, channel_axis = 3):
        super().__init__()
        self.weights_ = weights
        # First Block
        self.conv1 = Conv2D(64, (7,7),
                strides=(2,2), name='conv1', padding='same')
        self.bn1 = BatchNormalization(axis=channel_axis, name='bn1')

        # Second Block
        self.conv2 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv2')
        self.bn2 = BatchNormalization(axis=channel_axis, epsilon=0.00001, name='bn2')

        # Third Block
        self.conv3 = Conv2D(192, (3, 3), strides=(1, 1), padding='same', name='conv3')
        self.bn3 = BatchNormalization(axis=channel_axis, epsilon=0.00001, name='bn3')
        
    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.conv1.set_weights(self.weights_['conv1'])
        out_shape = self.conv1.compute_output_shape(input_shape)

        self.bn1.build(out_shape)
        self.bn1.set_weights(self.weights_['bn1'])
        out_shape = self.bn1.compute_output_shape(out_shape)
    
        self.conv2.build(out_shape)
        self.conv2.set_weights(self.weights_['conv2'])
        out_shape = self.conv2.compute_output_shape(out_shape)

        self.bn2.build(out_shape)
        self.bn2.set_weights(self.weights_['bn2'])
        out_shape = self.bn2.compute_output_shape(out_shape)

        self.conv3.build(out_shape)
        self.conv3.set_weights(self.weights_['conv3'])
        out_shape = self.conv3.compute_output_shape(out_shape)

        self.bn3.build(out_shape)
        self.bn3.set_weights(self.weights_['bn3'])

    def call(self, input_x):
        # First Block
        X = self.conv1(input_x)
        X = self.bn1(X)
        X = Activation('relu', name='relu1')(X)

        # MAXPOOL
        X = MaxPooling2D((3, 3), strides=2, padding='same', name="mxpool")(X)

        # Second Block
        X = self.conv2(X)
        X = self.bn2(X)
        X = Activation('relu', name='relu2')(X)

        # Third Block
        X = self.conv3(X)
        X = self.bn3(X)
        X = Activation('relu', name="relu3")(X)

        # MAXPOOL
        X = MaxPooling2D(pool_size=3, strides=2, padding='same', name='mxpool2')(X)
        
        return X


##################### Ending layer #########################
class EndingLayer(layers.Layer):
    def __init__(self, weights):
        super().__init__()
        self.weights_ = weights
        self.dense = Dense(128, name='dense_layer')
        
    def build(self, input_shape):
        self.dense.build(input_shape)
        self.dense.set_weights(self.weights_['dense_layer'])
        
    def call(self, input_x):
        X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(input_x)
        X = Flatten()(X)
        X = Dense(128, name='dense_layer')(X)

        # L2 normalization
        X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)
        return X


##################### Inception Model ######################
class Inception(models.Model):
    def __init__(self, weight_path):
        super().__init__()
        weights = np.load(weight_path, allow_pickle=True).item()
        # channel last configuration
        self.channel_axis = 3
        
        self.start_layer = StartingLayer(weights, channel_axis=self.channel_axis)
        # Inception 1: a/b/c
        self.inception_3a = InceptionLayer(weights, _1x1=64, _3x3r=96, _3x3=128, _5x5r=16, _5x5=32,
                                  _pool_conv=32, pooling='max', pool_stride=(2, 2), name='inception_3a')
        self.inception_3b = InceptionLayer(weights, _1x1=64, _3x3r=96, _3x3=128, _5x5r=32, _5x5=64,
                                  _pool_conv=64, pooling='avg', pool_stride=(3, 3), name='inception_3b')
        self.inception_3c = InceptionLayer(weights, _1x1=0, _3x3r=128, _3x3=256, _5x5r=32, _5x5=64, _pool_conv=0,
                                  pooling='max', strides=(2, 2), pool_stride=(2, 2), name='inception_3c')
        # Inception 2: a/e
        self.inception_4a = InceptionLayer(weights, _1x1=256, _3x3r=96, _3x3=192, _5x5r=32, _5x5=64, _pool_conv=128, 
                                  pooling='avg', pool_stride=(3, 3), name='inception_4a')
        self.inception_4e = InceptionLayer(weights, _1x1=0, _3x3r=160, _3x3=256, _5x5r=64, _5x5=128, _pool_conv=0,
                                  pooling='max', strides=(2, 2), pool_stride=(2, 2), name='inception_4e')
        # Inception 3: a/b
        self.inception_5a = InceptionLayer(weights, _1x1=256, _3x3r=96, _3x3=384, _5x5r=0, _5x5=0,
                                  _pool_conv=96, pooling='avg', pool_stride=(3, 3), name='inception_5a')
        self.inception_5b = InceptionLayer(weights, _1x1=256, _3x3r=96, _3x3=384, _5x5r=0, _5x5=0,
                                  _pool_conv=96, pooling='max', pool_stride=(2, 2), name='inception_5b')
        
        self.end_layer = EndingLayer(weights)
        
    def call(self, input_x):
        x = self.start_layer(input_x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        x = self.inception_4a(x)
        x = self.inception_4e(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.end_layer(x)
        return x

    
    def summary(self):
        x = Input((96, 96, 3))
        model = models.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


