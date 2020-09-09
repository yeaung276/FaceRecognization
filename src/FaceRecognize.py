from os.path import isfile
import scipy.io
import numpy as np
from keras import backend as K
from InceptionV2 import InceptionModuleBuilder, InceptionModel
from Utils import LAYERS, load_weights, img_to_encoding

K.set_image_data_format('channels_last')

print('loading weight matrix...', end='')
weight_matrix = np.load('net_weights.npy', allow_pickle=True).item()
print('ok')


class FaceRecognizer:

    def __init__(self, target_database_path='data_base.mat'):
        builder = InceptionModuleBuilder()
        self.database_path = target_database_path
        self.model = builder.BuildInception()
        self._load_weights_to_model()
        self._load_database(target_database_path)

    def _load_database(self, path):
        if isfile(path):
            self.target_database = scipy.io.loadmat(path)
        elif isfile('data_base.mat'):
            self.target_database = scipy.io.loadmat('data_base.mat')
        else:
            self.target_database = {}

    def _save_database(self, path):
        if isfile(path):
            scipy.io.savemat(path, self.target_database)
        else:
            scipy.io.savemat('data_base.mat', self.target_database)

    def _load_weights_to_model(self):
        for name in LAYERS:
            self.model.get_layer(name).set_weights(weight_matrix[name])

    def show_summary(self):
        self.model.summary()

    def add_target(self, image_path, target_name):
        enc = img_to_encoding(image_path, self.model)
        self.target_database.update({target_name: enc})
        self._save_database(self.database_path)

    def calculate_distance(self, img1, img2):
        enc1 = img_to_encoding(img1, self.model)
        enc2 = img_to_encoding(img2, self.model)
        return np.linalg.norm(enc1-enc2)

    # TODO: add calculate distance and find target method
