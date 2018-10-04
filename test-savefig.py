from brian2 import second
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2

MNIST_TRAIN_SPIKE_FILE = "First100ImgSpikeTrainData.h5"


def load_train_data(hdf5_path):
    f = h5py.File(hdf5_path)
    indices = f["indices"][:]
    times = f["times"][:]
    f.close()
    return indices, times


# indices, times = load_train_data(MNIST_TRAIN_SPIKE_FILE)
# times = times / (1000 * second)
# plt.plot(times, indices, "|")
# plt.xlim(20.0,20.25)
# plt.ylim(100,700)
# plt.savefig('test.png', dpi=1000)
# plt.show()

f = h5py.File("stdp_weights_100pics.h5")
w = f["weights"][:]
w = w / np.max(w)
f.close()
M = w.reshape(784,100)
M = M.transpose()

cv2.namedWindow('Weights', cv2.WINDOW_NORMAL)
for i in range(100):
    cv2.imshow('Weights', M[i].reshape((28, 28)))
    cv2.waitKey(0)