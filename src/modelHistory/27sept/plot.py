import scipy.io
from Utils.Utils import plot_history

mat = scipy.io.loadmat('0.1.0_hist.mat')
plot_history(mat)
