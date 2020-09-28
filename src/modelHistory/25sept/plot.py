import scipy.io
from Utils.Utils import plot_history

mat = scipy.io.loadmat('version3_hist.mat')
plot_history(mat)
