from contextlib import redirect_stdout
import scipy.io
from Utils import load_weights


def log(function, file='log.txt'):
    with open(file, 'w+') as f:
        with redirect_stdout(f):
            function()


def create_weight_mat():
    mat = load_weights()
    scipy.io.savemat('net_weights.mat', mat)


create_weight_mat()

