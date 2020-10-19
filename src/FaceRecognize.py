from os.path import isfile
import pickle
import numpy as np
from keras import backend as K
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model

from Model.SimeseNet import ModelBuilder
from Utils.inception_utils import img_to_encoding
from FaceDetect import FaceDetector

K.set_image_data_format('channels_last')


class FaceRecognizer:

    def __init__(self, target_database_path='Resources/faces/database.pkl'):
        self.database_path = target_database_path
        self.model, self.base_model = self._get_models()
        self.N = self._load_database(target_database_path)
        self.threadshold = None
        self.encs = None
        self.names = None

    def _load_database(self, path):
        if isfile(path):
            with open(path, 'rb') as f:
                self.target_database = pickle.load(f)
        else:
            self.target_database = {}
        return len(self.target_database.keys())

    def _save_database(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.target_database, f, pickle.HIGHEST_PROTOCOL)

    def show_summary(self):
        self.model.summary()
        self.base_model.summary()

    def add_target(self, image_path, target_name):
        fd = FaceDetector()
        _, face, _ = fd.detectFaces(image_path)
        assert len(face) == 1, "multiple face detected"

        enc = img_to_encoding(face[0], self.base_model)
        self.target_database.update({target_name: enc})
        self._save_database(self.database_path)

    def remove_target(self, target_name):
        del self.target_database[target_name]
        self._save_database(self.database_path)

    def predict(self, image_path1, image_path2):
        model = ModelBuilder.buildModel()
        model.load_weights('Resources/model/model.h5')

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
        embedding = model.predict_on_batch([x_1, x_2])
        return embedding

    @staticmethod
    def _get_models():
        model = ModelBuilder.buildModel()
        model.load_weights('Resources/model/model.h5')

        base = Model(model.get_layer('model').input, model.get_layer('model').output)

        upper = Model(model.get_layer('upper_model').input, model.get_layer('upper_model').output)

        return upper, base

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
            processed_encoding.append(np.vstack([enc]*self.N))
        return processed_encoding

    def start(self, confident_threadshold=0.5):
        self.threadshold = confident_threadshold
        names = []
        encs = []
        for name, enc in self.target_database.items():
            names.append(name)
            encs.append(enc)
        self.encs = np.vstack(encs)
        self.names = names

    def find(self, encoding):
        return self.model.predict_on_batch([encoding, self.encs])

    def find_on_batch(self, encodings):
        results = []
        names = []
        for enc in encodings:
            result = self.find(enc)
            name, _ = self.get_names(result)
            names.append(name)
            results.append(result)
        return names, np.hstack(results)

    def get_names(self, prop_matrix):
        p = np.asscalar(np.max(prop_matrix, axis=0))
        index = np.asscalar(np.argmax(prop_matrix, axis=0))
        if p > self.threadshold:
            name = self.names[index]
        else:
            name = 'unrecognized'
        return name, p

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
