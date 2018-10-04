from brian2 import *
import brian2.numpy_ as np
import brian2genn
import h5py
import time
import os

# global Params
input_layer_size = 784
hidden_layer_size = 100
output_layer_size = 10
train_pics = 5000
test_pics = 500
test_interval = 200
simulate_time = 50

class SNN():

    def __init__(self):
        self.i_layer_size = input_layer_size
        self.h_layer_size = hidden_layer_size
        self.o_layer_size = output_layer_size
        self.test_step = 8
        self.time_step = 1.0
        self.base_frequency = 250
        self.learnrate = 0.0005
        self.cita_h = 0.9
        self.cita_o = 0.025*self.h_layer_size
        self.reset = 0
        # set_device('genn')

    def train(self, sim_time, img_array, label, i_h_weight, h_o_weight):
        print("Current label is : %d" % label)
        start_scope()

        # really shit! the self.params connot be directly used in this part
        # Thus, we need to refer them before we use
        cita_h = self.cita_h
        cita_o = self.cita_o
        reset = self.reset
        time_step = self.time_step

        # define the neuron model
        eqs = ''' v :1 '''

        # the input
        rate = [np.float(x) / 255.0 * self.base_frequency for x in img_array]
        i_layer = PoissonGroup(self.i_layer_size, rate*Hz)

        # initialize the weights matrix to normal population (mu=0, sigma=1)
        # and we need to reshape the matrix to one-dim to match the conn.w
        weight_ih = i_h_weight
        weight_ho = h_o_weight

        # define the layer connection
        h_layer = NeuronGroup(self.h_layer_size, eqs, threshold='v>cita_h', reset='v=reset', method='exact')
        o_layer = NeuronGroup(self.o_layer_size, eqs, threshold='v>cita_o', reset='v=reset', method='exact')
        conn_ih = Synapses(i_layer, h_layer, model='w:1', on_pre='v_post += w')
        conn_ih.connect(p=1)
        conn_ih.w = weight_ih
        conn_ho = Synapses(h_layer, o_layer, model='w:1', on_pre='v_post += w')
        conn_ho.connect(p=1)
        conn_ho.w = weight_ho

        spike_i = SpikeMonitor(i_layer)
        spike_h = SpikeMonitor(h_layer)
        spike_o = SpikeMonitor(o_layer)

        for step in range(int(sim_time/time_step)):
            start = time.time()

            spike_i_last = array(spike_i.count).copy()
            spike_h_last = array(spike_h.count).copy()
            spike_o_last = array(spike_o.count).copy()

            run(time_step*ms)

            # count spikes of input & hidden layer in one timestep
            spike_sum_i = subtract(array(spike_i.count).copy(), array(spike_i_last))
            spike_sum_h = subtract(array(spike_h.count).copy(), array(spike_h_last))
            spike_sum_o = subtract(array(spike_o.count).copy(), array(spike_o_last))

            # compute ksai_o
            ksai_o = zeros(self.o_layer_size)
            for id_o in range(self.o_layer_size):
                if(spike_sum_o[id_o] >= 1):
                    if(id_o != label):
                        ksai_o[id_o] = -1
                else:
                    if(id_o == label):
                        ksai_o[id_o] = 1
            print(" [%2d] Spikes Sum:(%3d,%3d,%3d)" %
                  (step,sum(spike_sum_i), sum(spike_sum_h), sum(spike_sum_o)),end=", ")
            print(spike_sum_o,end=", ")

            # update the weight from hidden to output layer
            delta_weight_ho = dot(spike_sum_h.reshape(self.h_layer_size, 1),
                                     ksai_o.reshape(1, self.o_layer_size))*self.learnrate
            conn_ho.w += delta_weight_ho.flatten()

            # update the weight from input to hidden layer
            derivatives_h = spike_sum_h.copy()
            derivatives_h[derivatives_h>0] = 1
            ksai_h = dot(array(conn_ho.w).reshape(self.h_layer_size,self.o_layer_size),
                            ksai_o) * derivatives_h
            delta_weight_ih = dot(spike_sum_i.reshape(self.i_layer_size, 1),
                                     ksai_h.reshape(1, self.h_layer_size)) * self.learnrate
            conn_ih.w += delta_weight_ih.flatten()
            end = time.time()
            print("time cost : %.2f" % (end-start))
        return (conn_ih.w, conn_ho.w)

    def test(self, imgs_t, i_h_weight, h_o_weight):
        start_scope()
        cita_h = self.cita_h
        cita_o = self.cita_o
        reset = self.reset
        test_step = self.test_step
        time_step = self.time_step
        eqs = ''' v :1 '''

        # the input
        rate = [np.float(x)/255.0 * self.base_frequency for x in imgs_t]
        i_layer_t = PoissonGroup(self.i_layer_size, rate * Hz)

        # define the layer connection
        h_layer_t = NeuronGroup(self.h_layer_size, eqs, threshold='v>cita_h', reset='v=reset', method='exact')
        o_layer_t = NeuronGroup(self.o_layer_size, eqs, threshold='v>cita_o', reset='v=reset', method='exact')
        conn_ih_t = Synapses(i_layer_t, h_layer_t, model='w:1', on_pre='v_post += w')
        conn_ih_t.connect(p=1)
        conn_ih_t.w = i_h_weight
        conn_ho_t = Synapses(h_layer_t, o_layer_t, model='w:1', on_pre='v_post += w')
        conn_ho_t.connect(p=1)
        conn_ho_t.w = h_o_weight
        spike_output = SpikeMonitor(o_layer_t)

        run(test_step*time_step*ms)

        network_label = np.argmax(spike_output.count)
        return network_label

    def save_weight(self,i_h_weight, h_o_weight):
        print(" # Saving weights...")
        f = h5py.File("WEIGHTS_TRAIN.h5","w")
        f["weight_1"] = i_h_weight
        f["weight_2"] = h_o_weight
        f.close()

    def load_weight(self,weight_file_name):
        print(" # Loading weights...")
        f = h5py.File(weight_file_name,'r')
        weights_1 = f["weight_1"][:]
        weights_2 = f["weight_2"][:]
        f.close()
        return (weights_1, weights_2)


if __name__ == "__main__":
    f = h5py.File("../nest/HDF5_MNIST_TRAIN.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()
    f = h5py.File("../nest/HDF5_MNIST_TEST.h5", 'r')
    test_img = f["img"][:]
    test_label = f["label"][:]
    f.close()
    snn = SNN()
    # initialize the weights
    if os.path.exists("WEIGHTS_TRAIN.h5"):
        print(" # Loading weights...")
        f = h5py.File("WEIGHTS_TRAIN.h5", 'r')
        ih_w = f["weight_1"][:]
        ho_w = f["weight_2"][:]
        f.close()
    else:
        print(" # Creating New Weights...")
        ih_w = np.random.randn(784*100)
        ho_w = np.random.randn(100*10)

    for i in range(train_pics):
        # (i_h_weight_list, h_o_weight_list) = snn.train(simulate_time, img[i].flatten(), label[i], ih_w, ho_w)
        # snn.save_weight(i_h_weight_list, h_o_weight_list)
        # update weights used in training
        # ih_w = i_h_weight_list
        # ho_w = h_o_weight_list

        if(i%test_interval==0):
            correct_num = 0
            base_index = int(np.random.random(1)[0] * 9990)
            for u in range(test_pics):
                test_index = base_index + u
                label_t = snn.test(test_img[test_index].flatten(), ih_w, ho_w)
                correct_num += int(label_t == test_label[test_index])
            print(" Accuracy is: %.3f" % (np.float(correct_num) / np.float(test_pics)))
            record_file = open('test_record.txt', 'a+')
            localtime = time.asctime(time.localtime(time.time()))
            record_file.writelines('%s , Accuarcy : %f\n' % (localtime,(np.float(correct_num) / np.float(test_pics)) ))
            record_file.close()
            exit(0)
