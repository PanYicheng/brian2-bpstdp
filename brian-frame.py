from brian2 import *
import numpy as np
import h5py

class SNN():

    def __init__(self):
        self.input_layer_size = 784
        self.hidden_layer_size = 100
        self.output_lay1er_size = 10
        self.episilon = 4
        self.time_step = 1
        self.base_frequency = 250
        self.learnrate = 0.005
        self.cita_h = 0.9
        self.cita_o = 0.025*self.hidden_layer_size
        self.reset = 0
        # define the neuron model
        self.eqs = '''
                v:1
                cita_h:1
                cita_o:1
                reset:1
                '''

    def create_snn(self):
        # start_scope()

        # all zero rate,just set to create snn
        rate = np.zeros(self.input_layer_size)
        self.i_layer = PoissonGroup(self.input_layer_size, rate * Hz)
        self.spikemon_i = SpikeMonitor(self.i_layer)

        # initialize the weights matrix to normal population (mu=0, sigma=1)
        # and we need to reshape the matrix to one-dim to match the conn.w
        weight_ih_initial = np.random.randn(self.input_layer_size*self.hidden_layer_size)
        weight_ho_initial = np.random.randn(self.hidden_layer_size*self.output_layer_size)

        # define the hidden layer
        self.h_layer = NeuronGroup(self.hidden_layer_size, self.eqs, threshold='v>cita_h', reset='v=reset', method='exact')
        self.spikemon_h = SpikeMonitor(self.h_layer)
        self.h_layer.cita_h = self.cita_h
        # define the output layer
        self.o_layer = NeuronGroup(self.output_layer_size, self.eqs, threshold='v>cita_o', reset='v=reset', method='exact')
        self.spikemon_o = SpikeMonitor(self.o_layer)
        self.o_layer.cita_o = self.cita_o

        self.conn_ih = Synapses(self.i_layer, self.h_layer, model='w:1', on_pre='v_post += w')
        self.conn_ih.connect(p=1)
        self.conn_ih.w = 'i*1000+j'
        print(np.array(self.conn_ih.w).reshape(self.input_layer_size,self.hidden_layer_size))
        self.conn_ho = Synapses(self.h_layer, self.o_layer, model='w:1', on_pre='v+=w')
        self.conn_ho.connect(p=1)
        self.conn_ho.w = weight_ho_initial


    def train(self,sim_time,img_array,label):
        print("Current label is : %d" % label)
        # analogy the input
        # rate = np.zeros(self.input_layer_size)
        rate = [np.float(x) / 255.0 * self.base_frequency for x in img_array] * hertz
        self.i_layer.rates = rate

        for step in range(int(1)):
            spike_i_count_last = self.spikemon_i.count
            spike_h_count_last = self.spikemon_h.count
            spike_o_count_last = self.spikemon_o.count

            run(40 * ms)

            print(np.subtract(self.spikemon_o.count,spike_o_count_last))



if __name__ == "__main__":
    f = h5py.File("../nest/HDF5_MNIST_TRAIN.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()
    snn = SNN()
    snn.create_snn()
    snn.train(100, img[1].flatten(), label[1])