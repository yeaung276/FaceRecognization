from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import sys

from Utils.inception_utils import load_weights

# Utils


def log(function, file='log.txt'):
    with open(file, 'w+') as f:
        with redirect_stdout(f):
            function()


def create_weight_mat():
    mat = load_weights()
    np.save('net_weights.npy', mat)


def show_random_sample(input1, input2, label, pair = 1):
    fig = plt.figure()
    plt.axis('off')
    row = pair
    column = 2
    for i in range(pair):
        index = np.random.choice(range(len(label)))
        fig_id = i * 2 +1
        fig.add_subplot(row, column, fig_id)
        plt.imshow(input1[index, :, :])
        plt.title('same' if label[index] == 1 else 'diff')
        plt.axis('off')
        fig.add_subplot(row, column, fig_id+1)
        plt.imshow(input2[index, :, :])
        plt.title('same' if label[index] == 1 else 'diff')
        plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def plot_history(mat, start=0, end=-1):

    fig, (loss, acc) = plt.subplots(2, 1)

    loss.plot(mat['loss'].flatten()[start:end], label='training loss')
    loss.plot(mat['val_loss'].flatten()[start:end], label='validation loss')
    loss.set_title('loss')
    loss.set_xlabel('epoch')
    loss.set_ylabel('loss')
    loss.legend()

    acc.plot(mat['accuracy'].flatten()[start:end], label='training accuracy')
    acc.plot(mat['val_accuracy'].flatten()[start:end], label='validation accuracy')
    acc.set_title('accuracy')
    acc.set_xlabel('epoch')
    acc.set_ylabel('accuracy')
    acc.legend()

    plt.show()

