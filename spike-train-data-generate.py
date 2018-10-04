from brian2 import *
import brian2.numpy_ as np
import h5py
import time
import os
from matplotlib import pyplot as plt


MNIST_TRAIN_HDF5_FILE = '../nest/HDF5_MNIST_TRAIN.h5'
MNIST_TEST_HDF5_FILE = '../nest/HDF5_MNIST_TEST.h5'

TRAIN_OUTPUT_FILE = "First100ImgSpikeTrainData.h5"
# TEST_OUTPUT_FILE = "../nest/HDF5_TEST_SPIKE_DATA.h5"

IMG_SIZE = 784
bin_size = 2*ms
p = 1*Hz*bin_size  # Firing rate per neuron: 1Hz
total = 50*second  # Total length of our stimulus of one img
pattern_length = 250*ms
repeat_every = pattern_length * 4
n_repetitions = int(total/repeat_every)

f = h5py.File(MNIST_TRAIN_HDF5_FILE, 'r')
train_img = f["img"][:]
train_label = f["label"][:]
f.close()

N = 100

global_indices = np.array([])
global_times = np.array([])

start_time = time.time()

for i in range(N):
    spikes = np.random.rand(IMG_SIZE, int(total / bin_size)) < p
    # create zero pattern first for we only fill nonzero position after
    pattern = np.zeros([IMG_SIZE, int(pattern_length / bin_size)])
    img_flatten = np.array(train_img[i]).flatten()

    # This parameter is gotten using several experiments
    rates = img_flatten / 255 * 125 * Hz

    inp = PoissonGroup(IMG_SIZE, rates)
    spikeMonitor = SpikeMonitor(inp)
    run(250 * ms, report="text")

    indices, times = spikeMonitor.it
    for j in range(np.array(indices).shape[0]):
        pattern[indices[j]][int(times[j] / bin_size)] = True
    for rep in np.arange(n_repetitions):
        spikes[:, rep * int(repeat_every / bin_size):rep * int(repeat_every / bin_size) + int(
            pattern_length / bin_size)] = pattern
    indices, time_bins = spikes.nonzero()
    # add a const shift to move the spikes time later,
    # every img produces spikes for (total) seconds
    time_bins += np.repeat(i * int(total / bin_size),time_bins.shape[0])
    # print("Shift Value: %d " % (i * int(total / bin_size)))
    times = time_bins * bin_size

    # plt.plot(times / second, indices, '.')
    # # Add lines to show the repeated stimulation windows
    # plt.vlines(np.arange(n_repetitions) * repeat_every / second, 0, 500, color='gray')
    # plt.vlines(np.arange(n_repetitions) * repeat_every / second + pattern_length / second, 0, 500, color='gray')
    # plt.xlim(i * 10, i*10+5)
    # plt.show()

    # remove the unit before appending
    global_indices = np.append(global_indices,indices)
    global_times = np.append(global_times,times / ms)

f = h5py.File(TRAIN_OUTPUT_FILE,'w')
f["indices"] = global_indices
f["times"] = global_times
f.close()

end_time = time.time()

print("Generating finished! Time used:%f" % (end_time - start_time))

# f = h5py.File(MNIST_TEST_HDF5_FILE, 'r')
# test_img = f["img"][:]
# test_label = f["label"][:]
# f.close()