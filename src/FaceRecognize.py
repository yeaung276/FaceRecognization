from os.path import isfile
import scipy.io
import numpy as np
from keras import backend as K
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model
from keras.layers import Input

from Model.SimeseNet import ModelBuilder
from Utils.inception_utils import LAYERS, img_to_encoding

K.set_image_data_format('channels_last')


class FaceRecognizer:

    def __init__(self, target_database_path='data_base.mat'):
        self.database_path = target_database_path
        self.model, self.base_model = self._get_models()
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

    def show_summary(self):
        self.model.summary()

    def add_target(self, image_path, target_name):
        enc = img_to_encoding(image_path, self.model)
        self.target_database.update({target_name: enc})
        self._save_database(self.database_path)

    def predict(self, image_path1, image_path2):
        img1 = cv2.imread(image_path1, 1)
        img1 = cv2.resize(img1, (96, 96))
        img1 = img1[..., ::-1]
        img1 = np.around(img1 / 255.0, decimals=12)
        x_1 = np.array([img1])

        img2 = cv2.imread(image_path2, 1)
        img2 = cv2.resize(img2, (96, 96))
        img2 = img2[..., ::-1]
        img2 = np.around(img2 / 255.0, decimals=12)
        x_2 = np.array([img2])
        embedding = self.model.predict_on_batch([x_1, x_2])
        return embedding

    @staticmethod
    def _get_models():
        model = ModelBuilder.buildModel()
        model.load_weights('data/model/model.h5')

        base = Model(model.get_layer('model').input, model.get_layer('model').output)
        # X1 = Input(shape=128)
        # X2 = Input(shape=128)
        #
        # model.get_layer('average').input = [X1, X2]
        # model.get_layer('subtract').input = [X1, X2]
        # upper = Model(model.get_layer('dense').input, model.get_layer('dense_2').output)

        return model, base

    @staticmethod
    def preprocess_images(faces):
        def mapFunction(x):
            assert x.shape == (96, 96, 3), 'dimension error: {}'.format(x.shape)
            x = x[..., ::-1]
            x = np.around(x / 255.0, decimals=12)
            return x
        return list(map(mapFunction, faces))

    def get_encodings(self, faces):
        processed_images = self.preprocess_images(faces)
        return self.base_model.predict_on_batch(np.array(processed_images))

    def process_encoding(self, encodings):
        processed_encoding = []
        for enc in encodings:
            processed_encoding.append(np.vstack([enc]*3))
        return processed_encoding

    def test(self):
        emb = self.predict(image_path1='test_images/test(jnd).jpeg', image_path2='test_images/test(jsp).jpg')
        img1 = mpimg.imread('test_images/test(jnd).jpeg')
        img2 = mpimg.imread('test_images/test(jsp).jpg')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img1)
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax2.imshow(img2)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        plt.figtext(0.5, 0.15, 'similarity probability: {}'.format(round(np.asscalar(emb), 4)), ha='center', fontsize=13)
        plt.savefig('jackCantEscape.jpg')
        plt.show()


# h = FaceRecognizer()
# h.test()
