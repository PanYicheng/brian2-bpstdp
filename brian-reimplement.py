from brian2 import *
import brian2.numpy_ as np
import h5py
import time
import os

# set_device('cpp_standalone',build_on_run=False)

train_pics = 10000
test_pics = 100
test_interval = 1000


class SNN():
    def __init__(self):
        self.input_layer_size = 784
        self.hidden_layer_size = 100
        self.output_layer_size = 10
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
        self.base_frequency = 250
        self.learn_rate = 0.005
        self.time_step = 8.0
        # make sure "test_steps * time_step = 20.0"
        self.test_steps = 5

        self.rates = zeros(self.input_layer_size)
        # input neurons
        inp = NeuronGroup(self.input_layer_size, 'v:1', method='exact', threshold='v>=1',
                          reset='v=0',name="input")
        def update_volt():
            inp.v += self.rates
        network_op = NetworkOperation(update_volt, dt=4.0 * ms)
        # hidden neurons
        hidden = NeuronGroup(self.hidden_layer_size, self.eqs, threshold="v>cita_h",
                             reset='v=reset', method="exact",name="hidden")
        hidden.cita_h = self.cita_h
        hidden.reset = self.reset
        output = NeuronGroup(self.output_layer_size, self.eqs, threshold='v>cita_o',
                             reset='v=reset', method='exact',name="output")
        output.cita_o = self.cita_o
        output.reset = self.reset


        conn_ih = Synapses(inp, hidden, model='w:1', on_pre='v_post += w',name="conn_ih")
        conn_ih.connect(p=1)
        conn_ih.w = np.random.randn(self.input_layer_size*self.hidden_layer_size)
        conn_ho = Synapses(hidden, output, model='w:1', on_pre='v_post += w',name="conn_ho")
        conn_ho.connect(p=1)
        conn_ho.w = np.random.randn(self.hidden_layer_size*self.output_layer_size)
        self.net = Network(conn_ih, conn_ho, network_op,
                           inp, hidden, output)
        # self.net.store("initial_weight")

    def set_input(self,img_array):
        self.rates = img_array / 255.0
        self.net.set_states({"input":{"v":zeros(self.input_layer_size)}})
        self.net.set_states({"hidden":{"v":zeros(self.hidden_layer_size)}})
        self.net.set_states({"output":{"v":zeros(self.output_layer_size)}})

    def train(self, sim_time, img_array, label):
        print("# Current label is : %d" % label)
        # analogy the input
        # rate = np.zeros(self.input_layer_size)
        self.set_input(img_array)

        spikemon_output = SpikeMonitor(self.net["output"], name='output_spikes')
        spikemon_hidden = SpikeMonitor(self.net["hidden"], name='hidden_spikes')
        spikemon_input = SpikeMonitor(self.net["input"], name='input_spikes')
        spikemon_list = [spikemon_input, spikemon_hidden, spikemon_output]
        self.net.add(spikemon_list)

        input_spike_count = array(spikemon_input.count).copy()
        hidden_spike_count = array(spikemon_hidden.count).copy()
        output_spike_count = array(spikemon_output.count).copy()

        conn_ho_w = self.net.get_states()["conn_ho"]["w"]
        conn_ih_w = self.net.get_states()["conn_ih"]["w"]

        for ct in range(int(sim_time / self.time_step)):
            start_time = time.time()

            current_input_spike_count = spikemon_input.count
            current_hidden_spike_count = spikemon_hidden.count
            current_output_spike_count = spikemon_output.count

            self.net.run(self.time_step * ms)

            spike_sum_input = subtract(current_input_spike_count , input_spike_count)
            spike_sum_hidden = subtract(current_hidden_spike_count , hidden_spike_count)
            spike_sum_output = subtract(current_output_spike_count , output_spike_count)
            print("  [%2d] Spikes Sum:(%3d,%3d,%3d)" %
                  (ct,sum(spike_sum_input), sum(spike_sum_hidden), sum(spike_sum_output)),end=", ")
            print(spike_sum_output,end=", ")
            input_spike_count = array(current_input_spike_count).copy()
            hidden_spike_count = array(current_hidden_spike_count).copy()
            output_spike_count = array(current_output_spike_count).copy()

            ksai_output = zeros(self.output_layer_size)
            for _ in range(self.output_layer_size):
                if((spike_sum_output[_] >= 1) and (_ != label)):
                    ksai_output[_] = -1
                if((spike_sum_output[_] == 0) and (_ == label)):
                    ksai_output[_] = 1
            # update weights from hidden to output
            delta_weight_ho = dot(spike_sum_hidden.reshape((self.hidden_layer_size,1)),
                                  ksai_output.reshape((1,self.output_layer_size))) * self.learn_rate
            conn_ho_w = self.net.get_states()["conn_ho"]["w"]

            # update weights from input to hidden
            derivatives_h = spike_sum_hidden.copy()
            derivatives_h[derivatives_h>0] = 1
            ksai_hidden = dot(array(conn_ho_w).reshape((self.hidden_layer_size,self.output_layer_size)),
                              ksai_output) * derivatives_h
            delta_weight_ih = dot(spike_sum_input.reshape((self.input_layer_size,1)),
                                  ksai_hidden.reshape((1,self.hidden_layer_size))) * self.learn_rate

            conn_ho_w = conn_ho_w + delta_weight_ho.flatten()
            self.net.set_states({"conn_ho": {"w": conn_ho_w}})

            conn_ih_w = self.net.get_states()["conn_ih"]["w"]
            conn_ih_w = conn_ih_w + delta_weight_ih.flatten()
            self.net.set_states({"conn_ih": {"w": conn_ih_w}})
            end_time = time.time()
            print("Time Cost:%f" % (end_time - start_time))
        self.net.remove(spikemon_list)
        return (conn_ih_w,conn_ho_w)

    def test(self,imgs,labels):
        spikemon_output = SpikeMonitor(self.net["output"], name='output_spikes')
        spikemon_hidden = SpikeMonitor(self.net["hidden"], name='hidden_spikes')
        spikemon_input = SpikeMonitor(self.net["input"], name='input_spikes')
        spikemon_list = [spikemon_input,spikemon_hidden,spikemon_output]
        self.net.add(spikemon_list)

        num = len(labels)
        correct_num = 0
        last_count = np.zeros(self.output_layer_size)
        for _ in range(num):
            self.set_input(array(imgs[_]).flatten())
            self.net.run(self.test_steps * self.time_step * ms)
            current_count = self.net.get_states()["output_spikes"]["count"]
            increased_count = np.subtract(current_count,
                                             last_count)
            last_count = current_count
            correct_num += int(labels[_] == np.argmax(increased_count))
            print(increased_count,end=", ")
            print(labels[_])
        # device.build(directory='output', compile=True, run=True, debug=False)
        accuarcy = float(correct_num) / num
        print("# Accuracy:%f" % accuarcy)
        record_file = open('test_record_reim.txt', 'a+')
        localtime = time.asctime(time.localtime(time.time()))
        record_file.writelines('%s , Accuarcy : %f\n' % (localtime,accuarcy) )
        record_file.close()
        self.net.remove(spikemon_list)
        print(self.net)

    def save_weight(self,file_name):
        print("# Saving weights")
        f = h5py.File(file_name,'w')
        f["weight_1"] = self.net.get_states()["conn_ih"]["w"]
        f["weight_2"] = self.net.get_states()["conn_ho"]["w"]
        f.close()

    def load_weight(self,file_name):
        print("# Loading weights from %s" % file_name)
        f = h5py.File(file_name, 'r')
        self.net.set_states({"conn_ih":{"w":f["weight_1"][:]}})
        self.net.set_states({"conn_ho":{"w":f["weight_2"][:]}})
        f.close()


if __name__ == "__main__":
    f = h5py.File("./HDF5_MNIST_TRAIN.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()

    f = h5py.File("./HDF5_MNIST_TEST.h5", 'r')
    test_img = f["img"][:]
    test_label = f["label"][:]
    f.close()

    snn = SNN()
    if(os.path.exists("WEIGHTS_TRAIN_FULL.h5")):
        snn.load_weight("WEIGHTS_TRAIN_FULL.h5")
    for index in range(train_pics):
        snn.train(200, img[index].flatten(), label[index])
    #    snn.save_weight("WEIGHTS_TRAIN_REIM.h5")
        if(index % test_interval == 0):
            snn.save_weight("WEIGHTS_TRAIN_FULL.h5")
            start_index = int(np.random.rand(1)[0] * (10000-test_pics))
            snn.test(test_img[start_index:start_index+test_pics],test_label[start_index:start_index+test_pics])

    # base_index = int(np.random.rand(1)[0] * 9000)
    base_index = 0
    start_time = time.time()
    snn.test(test_img[base_index:base_index+10000],test_label[base_index:base_index+10000])
    end_time = time.time()
    print("Used Test Time:%f" % (end_time-start_time))
