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

IMG_SIZE = 784
total = 1000 * second
pattern_length = 250*ms
repeat_every = pattern_length * 4
n_repetitions = int(total/repeat_every)

train_data_total_time = 5000 * second
sim_time = train_data_total_time


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
# 设置是否进行训练，即权重是否会随着stdp的过程改变
learningRate = 1.0
if( learningRate > 0 ):
    print("# Train     Process:")
else:
    print("# Inference Process")

neurons = NeuronGroup(100, model=eqs, method='exact',
                      threshold='V > V_th', reset='V = V_r',
                      refractory=5*ms, name="output")
neurons.V = V_r

N_e = N
sgg_spikes = SpikeGeneratorGroup(N, indices, times, period=train_data_total_time, name="sgg_layer")

e_synapses = Synapses(sgg_spikes, neurons,
                    '''w : volt
                       dpre_trace/dt = -pre_trace / tau_trace : volt (event-driven)
                       dpost_trace/dt = -post_trace / tau_trace : volt (event-driven)''',
                       on_pre='''I_e += lambda_e*w
                                 pre_trace += A_pot
                                 w = clip(w + post_trace * learningRate, 0, w_max)''',
                       on_post='''post_trace += A_dep
                                  w = clip(w + pre_trace * learningRate, 0, w_max)''')
e_synapses.connect()

if_exist = os.path.exists(STDP_WEIGHTS_FILE)
# if_exist = False
if(if_exist):
    print("Loading existed weights")
    f = h5py.File(STDP_WEIGHTS_FILE)
    weights = f["weights"][:]
    f.close()
    e_synapses.w = weights * mV
else:
    print("Initating new weights")
    # adjust initiated weights to force output spike
    e_synapses.w = 'rand()**4 * 2*mV'

# mon = StateMonitor(neurons, 'V', record=0)

spike_mon = SpikeMonitor(neurons)

run(sim_time, report='stdout')

weights = np.array(e_synapses.w / mV)

f = h5py.File(STDP_WEIGHTS_FILE,'w')
f["weights"] = weights
f.close()

fig, ax = plt.subplots()
ax.hist(e_synapses.w/mV, bins=50)
ax.set(xlabel='$w_e$ (mV)')

fig, ax = plt.subplots()
ax.vlines(np.arange(n_repetitions*10)*repeat_every/second, 0, 100, color='gray', alpha=0.5)
ax.vlines(np.arange(n_repetitions*10)*repeat_every/second + pattern_length/second, 0, 100, color='gray', alpha=0.5)
ax.plot(spike_mon.t/second, spike_mon.i, '|')
ax.set(xlim=(0, 10), xlabel='time (s)')

plt.show()



