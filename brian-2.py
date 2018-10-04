from brian2 import *
import brian2genn
import numpy as np
import h5py
import time

class SNN():

    def __init__(self):
        self.i_layer_size = 784
        self.h_layer_size = 100
        self.o_layer_size = 10
        self.time_step = 0.2
        self.base_frequency = 250
        self.learnrate = 0.005
        self.cita_h = 0.9
        self.cita_o = 0.025*self.h_layer_size
        self.reset = 0

    def train(self, sim_time, img_array, label):
        print("Current label is : %d" % label)
        start = time.time()
        start_scope()

        # really shit! the self.params connot be directly used in this part
        # Thus, we need to refer them before we use
        cita_h = self.cita_h
        cita_o = self.cita_o
        reset = self.reset
        time_step = self.time_step

        # define the neuron model
        eqs = '''
        v :1
        '''

        # the input
        rate = [np.float(x) / 255.0 * self.base_frequency for x in img_array]
        i_layer = PoissonGroup(self.i_layer_size, rate*Hz)

        # initialize the weights matrix to normal population (mu=0, sigma=1)
        # and we need to reshape the matrix to one-dim to match the conn.w
        weight_ih_initial = np.random.randn(self.i_layer_size, self.h_layer_size).reshape(self.i_layer_size*self.h_layer_size)
        weight_ho_initial = np.random.randn(self.h_layer_size, self.o_layer_size).reshape(self.h_layer_size*self.o_layer_size)

        # define the layer connection
        h_layer = NeuronGroup(self.h_layer_size, eqs, threshold='v>cita_h', reset='v=reset', method='exact')
        o_layer = NeuronGroup(self.o_layer_size, eqs, threshold='v>cita_o', reset='v=reset', method='exact')
        conn_ih = Synapses(i_layer, h_layer, model='w:1', on_pre='v+=w')
        conn_ih.connect(p=1)
        conn_ih.w = weight_ih_initial
        conn_ho = Synapses(h_layer, o_layer, model='w:1', on_pre='v+=w')
        conn_ho.connect(p=1)
        conn_ho.w = weight_ho_initial

        for step in range(int(sim_time/time_step)):
            spike_i = SpikeMonitor(i_layer)
            spike_h = SpikeMonitor(h_layer)
            spike_o = SpikeMonitor(o_layer)
            spike_i_last = SpikeMonitor(i_layer).count
            spike_h_last = SpikeMonitor(h_layer).count
            spike_o_last = SpikeMonitor(o_layer).count

            run(time_step*ms)
            print(spike_o.count - spike_o_last)

            # compute ksai_o
            ksai_o = np.zeros(self.o_layer_size)
            for id_o in range(self.o_layer_size):
                spike_sum_o = spike_o.count[id_o] - spike_o_last[id_o]
                if(spike_sum_o >= 1):
                    if(id_o != label):
                        ksai_o[id_o] = -1
                else:
                    if(id_o == label):
                        ksai_o[id_o] = 1

            # count spikes of input & hidden layer in one timestep
            spike_sum_i = np.zeros(self.i_layer_size)
            for id_i in range(self.i_layer_size):
                spike_sum_i[id_i] = spike_i.count[id_i] - spike_i_last[id_i]
            spike_sum_h = np.zeros(self.h_layer_size)
            for id_h in range(self.h_layer_size):
                spike_sum_h[id_h] = spike_h.count[id_h] - spike_h_last[id_h]

            # update the weight from hidden to output layer
            delta_weight_ho = np.dot(spike_sum_h.reshape(self.h_layer_size, 1),
                                     ksai_o.reshape(1, self.o_layer_size))*self.learnrate
            conn_ho.w += delta_weight_ho.flatten()

            # update the weight from input to hidden layer
            derivatives_h = spike_sum_h.copy()
            derivatives_h[derivatives_h>0] = 1
            ksai_h = np.dot(np.array(conn_ho.w).reshape(100,10), ksai_o) * derivatives_h
            delta_weight_ih = np.dot(spike_sum_i.reshape(self.i_layer_size, 1),
                                     ksai_h.reshape(1, self.h_layer_size) * self.learnrate)
            conn_ih.w += delta_weight_ih.flatten()

            end = time.time()
            print("train for one step cost : %.2f" % (end-start))
        return (conn_ih.w, conn_ho.w)



if __name__ == "__main__":
    f = h5py.File("../nest/HDF5_MNIST_TRAIN.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()
    snn = SNN()
    snn.train(1, img[1].flatten(), label[1])
