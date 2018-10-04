from brian2 import *
import brian2.numpy_ as np
import brian2genn
import h5py

prefs["devices.genn.path"] = "/opt/pyc/genn-3.1.1"
prefs["devices.genn.cuda_path"] = "/usr/local/cuda"
set_device("cpp_standalone")

f = h5py.File("../nest/HDF5_MNIST_TRAIN.h5",'r')
img = f["img"][:]
label = f["label"][:]
f.close()
eqs = '''
    v:1
    cita_h:1
    cita_o:1
    reset:1
    '''

rates = array(img[0]).flatten() * Hz

inp = PoissonGroup(784,rates=rates,name='input')
hidden = NeuronGroup(100,eqs,threshold='v>cita_h',reset="v=0",method="euler",name="hidden")
hidden.cita_h = 0.9
output = NeuronGroup(10,eqs,threshold='v>cita_o',reset='v=0',method='euler',name="output")
output.cita_o = 0.025*100

conn_ih = Synapses(inp, hidden, model='w:1', on_pre='v_post += w',name="conn_ih")
conn_ih.connect(p=1)
conn_ih.w = np.random.randn(784*100)
conn_ho = Synapses(hidden, output, model='w:1', on_pre='v_post += w',name="conn_ho")
conn_ho.connect(p=1)
conn_ho.w = np.random.randn(100*10)

print("# Loading weights")
f = h5py.File("WEIGHTS_TRAIN.h5", 'r')
conn_ih.set_states({"w":f["weight_1"][:]})
conn_ho.set_states({"w":f["weight_2"][:]})
f.close()


spikemon = SpikeMonitor(output, name="spikemon")
run(100*ms, report="text")

count = spikemon.count
print(count)
predict = np.argmax(count)

print("Pre:%d,Real:%d" % (predict,label[0]))

# xlabel("Time(ms)")
# ylabel("Poisson spikes")
# plot(x,y,"ok")
# show()