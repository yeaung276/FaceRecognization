from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt

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
