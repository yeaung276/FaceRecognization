from keras.layers import Subtract, Average, Dropout
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import Flatten, Input, concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
import numpy as np

from Model.InceptionV2 import InceptionModuleBuilder
from Utils.inception_utils import LAYERS

K.set_image_data_format('channels_last')
##############################################################################################################

trainable_layers = ['dense_layer']

print('loading weight matrix...', end='')
weight_matrix = np.load('net_weights.npy', allow_pickle=True).item()
print('ok')

class ModelBuilder:

    @classmethod
    def _create_base_model(cls, print=False):
        facenet_builder = InceptionModuleBuilder()
        facenet = facenet_builder.BuildInception()
        cls._load_weights(facenet)
        for layer in facenet.layers:
            if layer.name not in trainable_layers:
                layer.trainable = False
        base_model = Model(inputs=facenet.input, outputs=facenet.get_layer('dense_layer').output)
        if print:
            base_model.summary()
        return base_model

    @staticmethod
    def _load_weights(model):
        for name in LAYERS:
            model.get_layer(name).set_weights(weight_matrix[name])

    @classmethod
    def buildModel(cls, print=False):
        base = cls._create_base_model()

        X1 = Input(shape=(96, 96, 3))
        X2 = Input(shape=(96, 96, 3))

        out1 = base(X1)
        out2 = base(X2)

        subtract = Subtract()([out1, out2])
        average = Average()([out1, out2])
        conc = concatenate([subtract, average], axis=1, name='dense_layer_1')

        X = Dropout(0.2)(conc)
        X = Dense(256, activation='relu')(X)
        X = Dense(256, activation='relu')(X)
        X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs=[X1, X2], outputs=X, name='SimeseFaceNet')
        if print:
            model.summary()
            plot_model(model)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def test(self):
        self._create_base_model(print=True)

