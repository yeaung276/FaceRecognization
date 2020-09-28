import numpy as np
import scipy.stats as ss
import os
import cv2
from Utils.Utils import show_random_sample
from Model.SimeseNet import ModelBuilder


#############################################################################################################
# Preparing Data

class _DATA_CONFIG:
    anchor = 10
    source = 'image'
    anchor_deviation = 1
    additional_anchor = 10
    additional_anchor_positive_chance = .4
    random_seed = 2


class Data:

    def __init__(self, ndarray, label):
        self.data = ndarray
        self.label = label

    def load(self):
        source = _DATA_CONFIG.source
        input1 = []
        input2 = []
        for pics in self.data:
            pic1 = os.path.join(source, pics[0])
            pic2 = os.path.join(source, pics[1])
            if os.path.getsize(pic1) != 0:
                img1 = self._read_image(pic1)
                input1.append(img1)
            if os.path.getsize(pic2) != 0:
                img2 = self._read_image(pic2)
                input2.append(img2)
        return np.array(input1), np.array(input2), self.label

    @staticmethod
    def _read_image(path):
        img1 = cv2.imread(path, 1)
        img1 = cv2.resize(img1, (96, 96))
        return img1

###################################################################


class GenerateDataset:

    def __init__(self, source='images'):
        np.random.seed(_DATA_CONFIG.random_seed)
        self.person_list = os.listdir(source)
        _DATA_CONFIG.source = source
        self.face_dictionary = {}
        self.match_data = []
        self.unmatch_data = []
        self.anchors = {}
        self.add_anchors = {}
        self.total_example = 0
        self.dataset = []
        self.labels = []
        for person in self.person_list:
            path = os.path.join(source, person)
            self.face_dictionary[person] = os.listdir(path)

    def _choose_anchors(self):
        dev = _DATA_CONFIG.anchor_deviation
        choice_range_mean = int(_DATA_CONFIG.anchor / len(self.person_list))
        no_of_anchor_per_person_choice = np.arange(choice_range_mean-dev, choice_range_mean+dev)

        xU, xL = no_of_anchor_per_person_choice + 0.5, no_of_anchor_per_person_choice - 0.5
        prob = ss.norm.cdf(xU, loc=choice_range_mean, scale=dev) - ss.norm.cdf(xL, loc=choice_range_mean, scale=dev)
        prob = prob / prob.sum()

        no_of_anchor_per_person = np.random.choice(no_of_anchor_per_person_choice, replace=True,
                                                   size=len(self.person_list), p=prob)

        no_of_anchor = np.sum(no_of_anchor_per_person)
        for i, name in enumerate(self.person_list):
            self.anchors[name] = np.random.choice(self.face_dictionary[name], size=no_of_anchor_per_person[i])
        return no_of_anchor

    def _choose_additional_anchor(self):
        anchor_per_person = int(_DATA_CONFIG.additional_anchor/len(self.person_list))
        dev = int(anchor_per_person * 0.1) + 1
        choice = np.arange(anchor_per_person - dev, anchor_per_person + dev)

        no_anchors = 0
        for name in self.person_list:
            num = np.random.choice(choice)
            no_anchors += num
            anchors = np.random.choice(self.face_dictionary[name], size=num)
            self.add_anchors[name] = anchors
        return  no_anchors

    def _find_match(self, name, size=1):
        choose = np.random.choice(self.face_dictionary[name], size=size)
        return choose, name

    def _find_unmatch(self, name):
        while True:
            choose_name = np.random.choice(self.person_list)
            if choose_name != name:
                break
        choice =  np.random.choice(self.face_dictionary[choose_name])
        return choice, choose_name

    def _create_unequal_pairs(self):
        pos = 0
        neg = 0
        for name, anchor in self.add_anchors.items():
            for i in anchor:
                if np.random.random(1) < _DATA_CONFIG.additional_anchor_positive_chance:
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

    def _match(self):
        no_match = 0
        for name, anchor in self.anchors.items():
            positive_match, _ = self._find_match(name, len(anchor))
            for i in range(len(anchor)):
                self.match_data.append(
                    (os.path.join(name, anchor[i]), os.path.join(name, positive_match[i]))
                )
                no_match += 1
        return no_match

    def _unmatch(self):
        no_unmatch = 0
        for name, anchor in self.anchors.items():
            for i in anchor:
                negative_match, choose_name = self._find_unmatch(name)
                self.unmatch_data.append(
                    (os.path.join(name, i), os.path.join(choose_name, negative_match))
                )
                no_unmatch += 1
        return  no_unmatch

    def generate_list(self):
        no_anchor = self._choose_anchors()
        no_match = self._match()
        no_unmatch = self._unmatch()
        no_add_anchor = self._choose_additional_anchor()
        pos, neg = self._create_unequal_pairs()
        tot = len(self.match_data)+len(self.unmatch_data)
        self.total_example = tot
        self.dataset.extend(self.match_data)
        self.labels.extend([1] * len(self.match_data))
        self.dataset.extend(self.unmatch_data)
        self.labels.extend([0] * len(self.unmatch_data))
        assert len(self.dataset) == self.total_example
        assert len(self.labels) == self.total_example

        print('Number of anchor created: {}\nNumber of positive examples: {}\nNumber of negative example: {}'
              '\nUnbalance pos: {}\nUnbalance neg: {},\nTotal: {}'.format(
                                no_anchor+no_add_anchor, no_match+pos, no_unmatch+neg, pos, neg, tot
                            ))

        return no_anchor, no_match, no_unmatch

    def shuffle(self):
        np.random.shuffle(self.match_data)
        np.random.shuffle(self.unmatch_data)

    def split(self, split_ratio=0.8):
        no_of_train_example = int(self.total_example * split_ratio)
        example_range = np.arange(0, self.total_example)

        training_examples, test_examples = np.split(np.random.permutation(example_range), [no_of_train_example])
        training_examples = training_examples.astype(int)
        test_examples = test_examples.astype(int)

        return Data(np.array(self.dataset)[training_examples], np.array(self.labels)[training_examples]), Data(
                np.array(self.dataset)[test_examples], np.array(self.labels)[test_examples])

    def get_list(self):
        return self.dataset, self.labels

###################################################################################################################


tt = GenerateDataset(source='images/Training')
tt.generate_list()
train, test = tt.split()
i1, i2, y = train.load()
print(i1.shape, i2.shape, y.shape)
show_random_sample(i1, i2, y, pair=5)

i1v, i2v, yv = test.load()
print(i1v.shape, i2v.shape, yv.shape)
show_random_sample(i1v, i2v, yv, pair=5)

model = ModelBuilder.buildModel()
model.fit([i1, i2], y, epochs=10, steps_per_epoch=100, validation_data=([i1v, i2v], yv))





