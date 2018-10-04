from brian2 import *
import brian2.numpy_ as np
import brian2genn
import time

prefs["devices.genn.path"] = "/opt/pyc/genn-3.1.1"
prefs["devices.genn.cuda_path"] = "/usr/local/cuda"
set_device("genn")

N = 10000  # 10000 "neurons"
bin_size = 2*ms
p = 1*Hz*bin_size  # Firing rate per neuron: 1Hz
total = 10*second  # Total length of our stimulus

spikes = np.random.rand(N, int(total/bin_size)) < p

pattern_length = 250*ms
pattern = spikes[:, 0:int(pattern_length/bin_size)]

repeat_every = pattern_length * 4
n_repetitions = int(total/repeat_every)

for rep in np.arange(n_repetitions):
    spikes[:, rep*int(repeat_every/bin_size):rep*int(repeat_every/bin_size)+int(pattern_length/bin_size)] = pattern

indices, time_bins = spikes.nonzero()
times = time_bins * bin_size

fig, ax = plt.subplots()
ax.plot(times/second, indices, '.')
# Add lines to show the repeated stimulation windows
ax.vlines(np.arange(n_repetitions)*repeat_every/second, 0, 500, color='gray')
ax.vlines(np.arange(n_repetitions)*repeat_every/second + pattern_length/second, 0, 500, color='gray')
# Restrict the plot to the first 500 neurons
ax.set(ylim=(0, 500))

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
neurons = NeuronGroup(100, model=eqs, method='exact',
                      threshold='V > V_th', reset='V = V_r',
                      refractory=5*ms)
neurons.V = V_r

N_e = 8000
N_i = 2000
N = N_e + N_i
spikes = SpikeGeneratorGroup(N, indices, times, period=total)

e_synapses = Synapses(spikes, neurons,
                    '''w : volt
                       dpre_trace/dt = -pre_trace / tau_trace : volt (event-driven)
                       dpost_trace/dt = -post_trace / tau_trace : volt (event-driven)''',
                       on_pre='''I_e += lambda_e*w
                                 pre_trace += A_pot
                                 w = clip(w + post_trace, 0, w_max)''',
                       on_post='''post_trace += A_dep
                                  w = clip(w + pre_trace, 0, w_max)''')
e_synapses.connect('i<N_e')
e_synapses.w = 'rand()**4 * 2*mV'

i_synapses = Synapses(spikes, neurons, on_pre='I_i += lambda_i*1*mV')
i_synapses.connect('i>=N_e')

# mon = StateMonitor(neurons, 'V', record=0)

spike_mon = SpikeMonitor(neurons)

# fig, ax = plt.subplots()
# ax.hist(e_synapses.w/mV, bins=50)
# ax.set(xlabel='$w_e$ (mV)')


start_time = time.time()
run(total*10, report='text')
end_time = time.time()
print("Used Time:%f" % (end_time - start_time))

# fig, ax = plt.subplots()
# ax.hist(e_synapses.w/mV, bins=50)
# ax.set(xlabel='$w_e$ (mV)')
#
# fig, ax = plt.subplots()
# ax.vlines(np.arange(n_repetitions*10)*repeat_every/second, 0, 100, color='gray', alpha=0.5)
# ax.vlines(np.arange(n_repetitions*10)*repeat_every/second + pattern_length/second, 0, 100, color='gray', alpha=0.5)
# ax.plot(spike_mon.t/second, spike_mon.i, '|')
# ax.set(xlim=(0, 5), xlabel='time (s)')
#
# time_indices = list(zip(spike_mon.t / second,spike_mon.i))
#
#
# def takeFirst(elem):
#     return elem[0]
#
#
# time_indices.sort(key=takeFirst)
#
# sum_spikes = np.zeros(shape=[200,100],dtype=np.int32)
#
# end_time = 0.25
# current_interval = 0
# for i in range(len(time_indices)):
#     if(time_indices[i][0] < end_time):
#         sum_spikes[current_interval][time_indices[i][1]] += 1
#     else:
#         end_time += 1.0
#         current_interval += 1
#
# for i in range(100):
#     print("Iteration: %d" % i, end=": ")
#     for j in range(100):
#         print(sum_spikes[i][j], end=", ")
#     print()
#
# max_spikes = np.max(sum_spikes)
#
# for i in range(0,100):
#     plt.subplot(100,1,i)
#     plt.bar(range(0,100),sum_spikes[i])
# plt.show()
