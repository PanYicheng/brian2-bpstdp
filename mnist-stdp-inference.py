from brian2 import *
import brian2.numpy_ as np
import h5py
import time
import os
import brian2genn

prefs["devices.genn.path"] = "/opt/pyc/genn-3.1.1"
prefs["devices.genn.cuda_path"] = "/usr/local/cuda"
set_device("genn")


MNIST_TRAIN_HDF5_FILE = '../nest/HDF5_MNIST_TRAIN.h5'
MNIST_TEST_HDF5_FILE = '../nest/HDF5_MNIST_TEST.h5'

MNIST_TRAIN_SPIKE_FILE = "First100ImgSpikeTrainData.h5"
STDP_WEIGHTS_FILE = "stdp_weights_100pics.h5"


INFERENCE_OUTPUT_FILE = "inference100pics.h5"

IMG_SIZE = 784
total = 1000 * second
pattern_length = 250*ms
repeat_every = pattern_length * 4
n_repetitions = int(total/repeat_every)

train_data_total_time = 5000 * second
sim_time = 50 * 10 * second


def load_train_data(hdf5_path):
    f = h5py.File(hdf5_path)
    indices = f["indices"][:]
    times = f["times"][:]
    f.close()
    return indices, times


indices, times = load_train_data(MNIST_TRAIN_SPIKE_FILE)
# attention, the generated times' unit is ms
times = times * ms

N = 784
tau_m = 5*ms
V_r = -70*mV
V_th = -55*mV
tau_e = 3*ms
tau_i = 10*ms
lambda_e = (tau_e / tau_m) ** (tau_m / (tau_e - tau_m))
lambda_i = - (tau_i / tau_m) ** (tau_m / (tau_i - tau_m))
tau_trace = 20*ms
w_max = 2*mV
A_pot = 0.02*w_max
A_dep = -1.2*A_pot
eqs = '''
dV/dt = ((V_r - V) + I_e + I_i)/ tau_m : volt (unless refractory)
dI_e/dt = -I_e/tau_e : volt
dI_i/dt = -I_i/tau_i : volt
'''

print("# Inference Process")

neurons = NeuronGroup(100, model=eqs, method='euler',
                      threshold='V > V_th', reset='V = V_r',
                      refractory=5*ms, name="output")
neurons.V = V_r

N_e = N
sgg_spikes = SpikeGeneratorGroup(N, indices, times, period=train_data_total_time, name="sgg_layer")

e_synapses = Synapses(sgg_spikes, neurons,
                    '''w : volt''',
                       on_pre='''I_e += lambda_e*w''',
                       on_post='''w = w''')
e_synapses.connect()

if_exist = os.path.exists(STDP_WEIGHTS_FILE)
# if_exist = False
if(if_exist):
    print("Loading existed weights")
    f = h5py.File(STDP_WEIGHTS_FILE)
    weights = f["weights"][:]
    f.close()
    e_synapses.w = weights * volt
else:
    print("[E] No weights file found!")
    exit(1)

# mon = StateMonitor(neurons, 'V', record=0)

spike_mon = SpikeMonitor(neurons)

run(sim_time, report='stdout')

output_indices = spike_mon.i
output_times = spike_mon.t / second

f = h5py.File(INFERENCE_OUTPUT_FILE)
f["indices"] = output_indices
f["times"] = output_times
f.close()

sum_spikes = np.zeros(shape=[10, 100], dtype=np.int32)
end_time = 0.25
current_index = 0
for i in range(len(output_times)):
    if(output_times[i] < end_time):
        if(output_times[i] >= end_time - 0.25):
            sum_spikes[current_index][output_indices[i]] += 1
    else:
        end_time += 50.0
        current_index += 1
        print(" Progress : %f " % (i / len(output_times)))

f = h5py.File(MNIST_TRAIN_HDF5_FILE)
train_label = f["label"][:]
f.close()

fig = plt.figure(figsize=(8,20))
for i in range(10):
    print("Index: %d, Number: %d" % (i,train_label[i]), end=": \n")
    for j in range(100):
        print(sum_spikes[i][j], end=", ")
    print()
    plt.subplot(5,2,i+1)
    plt.imshow(sum_spikes[i].reshape((10,10)))
    plt.hist(sum_spikes[i], bins=50)
    plt.title("Label {}".format(train_label[i]))
plt.show()

# def takeFirst(elem):
#     return elem[0]
#
#
# time_indices.sort(key = takeFirst)
# sum_spikes = np.zeros(shape=[10, 784], dtype=np.int32)
# end_time = 0.25
# current_index = 0
# for i in range(len(time_indices)):
#     if(time_indices[i][0] < end_time):
#         sum_spikes[current_index][time_indices[i][1]] += 1
#     else:
#         end_time += 50.0
#         current_index += 1
#
# f = h5py.File(MNIST_TRAIN_HDF5_FILE)
# train_label = f["label"][:]
# f.close()
#
# fig, ax = plt.subplots()
# ax.vlines(np.arange(n_repetitions*10)*repeat_every/second, 0, 100, color='gray', alpha=0.5)
# ax.vlines(np.arange(n_repetitions*10)*repeat_every/second + pattern_length/second, 0, 100, color='gray', alpha=0.5)
# ax.plot(spike_mon.t/second, spike_mon.i, '|')
# ax.set(xlim=(0, 10), xlabel='time (s)')
#
# plt.show()



