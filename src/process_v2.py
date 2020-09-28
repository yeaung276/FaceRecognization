import numpy as np
import scipy.stats as ss
import scipy.io
import os
import cv2
from keras.optimizers import Adam
from Utils.Utils import show_random_sample
from Model.SimeseNet import ModelBuilder
from Utils.Utils import Logger
import sys


#############################################################################################################
# Preparing Data

class _DATA_CONFIG:
    anchor = 10
    source = 'image'
    anchor_deviation = 10
    anchor_positive_chance = .4
    random_seed = 2


class Data:

    def __init__(self, ndarray, label, source=''):
        self.data = ndarray
        self.label = label
        self.source = source

    def load(self, channel_order='BGR'):
        input1 = []
        input2 = []
        for pics in self.data:
            pic1 = os.path.join(self.source, pics[0])
            pic2 = os.path.join(self.source, pics[1])
            if os.path.getsize(pic1) != 0:
                img1 = self._read_image(pic1, channel_order)
                input1.append(img1)
            if os.path.getsize(pic2) != 0:
                img2 = self._read_image(pic2, channel_order)
                input2.append(img2)
        return np.array(input1), np.array(input2), self.label

    @staticmethod
    def _read_image(path, channel_order):
        img1 = cv2.imread(path, 1)
        if channel_order == 'RGB':
            img1 = img1[..., ::-1]
        img1 = cv2.resize(img1, (96, 96))
        img1 = np.around(img1 / 255.0, decimals=12)
        return img1

###################################################################


class GenerateDataset:

    def __init__(self,  no_of_anchor, source='images',):
        np.random.seed(_DATA_CONFIG.random_seed)
        self._no_anchor = no_of_anchor
        self.person_list = os.listdir(source)
        self.source = source
        self.face_dictionary = {}
        self.anchors = {}
        self.match_data = []
        self.unmatch_data = []
        self.total_example = 0
        self.dataset = []
        self.labels = []
        for person in self.person_list:
            path = os.path.join(source, person)
            self.face_dictionary[person] = os.listdir(path)

    def _choose_anchors(self):
        count = 0
        while count != self._no_anchor:
            person = np.random.choice(self.person_list)
            if person not in self.anchors.keys():
                self.anchors[person] = [np.random.choice(self.face_dictionary[person])]
            else:
                self.anchors[person].append(np.random.choice(self.face_dictionary[person]))
            count = count + 1
        return count

    def _find_match(self, name, size=1):
        choose = np.random.choice(self.face_dictionary[name], size=size)
        return choose, name

    def _find_unmatch(self, name):
        while True:
            choose_name = np.random.choice(self.person_list)
            if choose_name != name:
                break
        choice = np.random.choice(self.face_dictionary[choose_name])
        return choice, choose_name

    def _create_unequal_pairs(self):
        pos = 0
        neg = 0
        for name, anchor in self.anchors.items():
            for i in anchor:
                if np.random.random(1) < _DATA_CONFIG.anchor_positive_chance:
                    pair, choose_name = self._find_match(name)
                    self.match_data.append(
                        (os.path.join(name, i), os.path.join(choose_name, pair[0]))
                    )
                    pos += 1
                else:
                    pair, choose_name = self._find_unmatch(name)
                    self.unmatch_data.append(
                        (os.path.join(name, i), os.path.join(choose_name, pair))
                    )
                    neg += 1
        return pos, neg

    def generate_list(self):
        no_anchor = self._choose_anchors()
        pos, neg = self._create_unequal_pairs()
        tot = len(self.match_data)+len(self.unmatch_data)
        self.total_example = tot
        self.dataset.extend(self.match_data)
        self.labels.extend([1] * len(self.match_data))
        self.dataset.extend(self.unmatch_data)
        self.labels.extend([0] * len(self.unmatch_data))
        assert len(self.dataset) == self.total_example
        assert len(self.labels) == self.total_example
        assert pos+neg == self.total_example

        print('Number of anchor created: {}\nNumber of positive examples: {}\nNumber of negative example: {}'.format(
            no_anchor, pos, neg, tot
        ))

        return no_anchor, pos, neg

    def getData(self):
        no_of_examples = np.arange(0, self.total_example)
        indexes = np.random.permutation(no_of_examples)
        return Data(np.array(self.dataset)[indexes], np.array(self.labels)[indexes], source=self.source)

    def get_list(self):
        return self.dataset, self.labels

    def test(self):
        self.generate_list()
        data = self.getData()
        print(data.data)
        print(data.label)

###################################################################################################################

# Training the network


def Train_Model():
    # hypermeters

    learning_rate = 0.00005
    steps_per_epochs = 200
    epochs = 500
    note =  'reduce complexity of the model' \
            'increase regularization parameter to 0.2' \
            'remove last layer of base to non-trainable'

    Train = GenerateDataset(5000, source='images/Training')
    Test = GenerateDataset(500, source='images/Testing')

    print('Training Data...')
    Train.generate_list()
    train = Train.getData()

    print('Test Data...')
    Test.generate_list()
    test = Test.getData()

    i1, i2, y = train.load(channel_order='RGB')
    print(i1.shape, i2.shape, y.shape)
    # show_random_sample(i1, i2, y, pair=5)

    i1v, i2v, yv = test.load()
    print(i1v.shape, i2v.shape, yv.shape)
    # show_random_sample(i1v, i2v, yv, pair=5)

    model = ModelBuilder.buildModel()
    print('hyperparameters')
    print('learning rate: {}\nSteps per Epoch: {}\nNo: of epochs: {}'.format(learning_rate, steps_per_epochs, epochs))
    print('change Note::')
    print(note)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    print('Training Progress')
    history = model.fit([i1, i2], y, epochs=epochs, steps_per_epoch=steps_per_epochs, validation_data=([i1v, i2v], yv))
    model.save('modelHistory/27sept/model.h5')
    scipy.io.savemat('modelHistory/27sept/0.0.1_hist.mat', history.history)


sys.stdout = Logger('modelHistory/27sept/0.0.1.txt')
Train_Model()





