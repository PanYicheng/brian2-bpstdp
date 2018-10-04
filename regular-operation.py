from brian2 import *
import brian2.numpy_ as np
import brian2genn
import h5py
set_device("genn")

# global Params
input_layer_size = 784
hidden_layer_size = 100
output_layer_size = 10
cita_h = 0.9
cita_o = 0.025*hidden_layer_size
reset = 0
test_time = 100

f = h5py.File("../nest/HDF5_MNIST_TEST.h5",'r')
img = f["img"][:]
label = f["label"][:]
f.close()

print(" # Loading weights...")
f = h5py.File("WEIGHTS_TRAIN_REIM.h5",'r')
weights_1 = f["weight_1"][:]
weights_2 = f["weight_2"][:]
f.close()

rates = array(img[1]).flatten()
rates = [float(x) / 255.0 for x in rates]
# print(sum(rates))

my_clock = Clock(dt=4.0*ms,name="clock")

# input neurons
inp = NeuronGroup(784,'v:1',method='exact',threshold='v>=1',reset='v=0',name="inp")
def update_volt():
    inp.v += rates
network_op = NetworkOperation(update_volt,clock=my_clock)
spikemon_inp = SpikeMonitor(inp,name="inp_spikes")
# hidden neurons
hidden = NeuronGroup(100,"v:1",method="exact",threshold="v>cita_h",reset='v=reset',name="hidden")
output = NeuronGroup(output_layer_size, "v:1", threshold='v>cita_o', reset='v=reset', method='exact')
spikemon = SpikeMonitor(output,name='output_spikes')

conn_ih = Synapses(inp, hidden, model='w:1', on_pre='v_post += w')
conn_ih.connect(p=1)
conn_ih.w = weights_1
conn_ho = Synapses(hidden, output, model='w:1', on_pre='v_post += w')
conn_ho.connect(p=1)
conn_ho.w = weights_2

net = Network(inp,network_op,hidden,output,spikemon,conn_ih,conn_ho,spikemon_inp,my_clock)




def test_one_pic(img_array,label,sim_time):
    global rates
    rates = img_array / 255.0
    net.run(sim_time*ms)
    count = net.get_states()["output_spikes"]["count"]
    print("prediction:%d,label:%d" % (argmax(count), label),end="; ")
    print(count)
    return (argmax(count) == label)


def test_200_pics():
    y = []
    x = []
    for sim_time in range(20,25):
        correct =0
        for i in range(200):
            correct += int(test_one_pic(array(img[i]).flatten(),label[i],sim_time))
        y.append(correct / 200.0)
        x.append(sim_time)
    plot(x,y,'ok')
    xlabel("Test Time(ms)")
    ylabel("Accuracy (100 pics)")
    show()

net.run(20*ms)


# test_one_pic(array(img[0]).flatten(),label[0],100)
# print(net.get_states()["inp_spikes"]["count"])

# test_200_pics()
# subplot(212)
# print(label[1])
# imshow(count.reshape((28,28)))
# show()

