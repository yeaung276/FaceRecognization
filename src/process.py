from contextlib import redirect_stdout
import numpy as np
import scipy.stats as ss
import os

from Utils import load_weights


def log(function, file='log.txt'):
    with open(file, 'w+') as f:
        with redirect_stdout(f):
            function()


def create_weight_mat():
    mat = load_weights()
    np.save('net_weights.npy', mat)


class _DATA_CONFIG:
    anchor = 1000
    anchor_deviation = 10
    additional_anchor = 500
    additional_anchor_positive_chance = .8
    random_seed = 0
    training_data_ratio = 0.8


class GenerateDataset:

    def __init__(self, source='images'):
        np.random.seed(_DATA_CONFIG.random_seed)
        self.person_list = os.listdir(source)
        self.face_dictionary = {}
        self.match_data = []
        self.unmatch_data = []
        self.anchors = {}
        self.add_anchors = {}
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
        return np.random.choice(self.face_dictionary[name], size=size), name

    def _find_unmatch(self, name):
        while True:
            choose_name = np.random.choice(self.person_list)
            if choose_name is not name:
                break
        return np.random.choice(self.face_dictionary[name]), choose_name

    def create_unequal_pairs(self):
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
        pos, neg = self.create_unequal_pairs()

        print('Number of anchor created: {}\nNumber of positive examples: {}\nNumber of negative example: {}'
              '\nUnbalance pos: {}\nUnbalance neg: {}'.format(
                            no_anchor+no_add_anchor, no_match+pos, no_unmatch+neg, pos, neg
                                    ))

        return no_anchor, no_match, no_unmatch

    def get_list(self):
        return self.match_data, self.unmatch_data



tt = GenerateDataset(source='images')
tt.generate_list()


