"""
    Helper function to generate mnist data from
    raw data to hdf5 for fast read
    Notes:
        mnist_dir: the dir where you stores the raw mnist file from lecun's web
            ['t10k-images-idx3-ubyte',
             'train-images-idx3-ubyte',
             'train-labels-idx1-ubyte',
             't10k-labels-idx1-ubyte']
"""
import time
import h5py
import os
import numpy as np

mnist_dir = "/data/mnist/"


def read_mnist_data(train_or_test: str):
    if train_or_test == "train":
        file_name_prefix = "train"
    else:
        file_name_prefix = "t10k"
    img_file = None
    try:
        img_file = open(mnist_dir + file_name_prefix + "-images-idx3-ubyte", "rb")
    except FileNotFoundError:
        print("Cannot Find Mnist Raw Data")
        exit(1)
    magic = img_file.read(4).hex()
    print("magic:", magic)
    if magic != '00000803':
        img_file.close()
        exit(1)
    img_num = eval('0x' + img_file.read(4).hex())
    print("img num:", img_num)
    mnist_height = eval('0x' + img_file.read(4).hex())
    mnist_width = eval('0x' + img_file.read(4).hex())
    print('height * width:', mnist_height, mnist_width)
    img_set = img_file.read(img_num * mnist_height * mnist_width)
    img_file.close()
    ret_img = []
    disp_img = np.zeros((mnist_height, mnist_width))
    for img_ct in range(img_num):
        one_img = img_set[(img_ct * mnist_height * mnist_width):((img_ct + 1) * mnist_height * mnist_width)]
        for _ in range(mnist_height * mnist_width):
            disp_img[_ // mnist_width][_ % mnist_width] = one_img[_]
        # print(disp_img)
        ret_img.append(disp_img.copy())

    label_file = open(mnist_dir + file_name_prefix + "-labels-idx1-ubyte", "rb")
    label_file.read(8)
    label_set = label_file.read(img_num)
    label_file.close()
    ret_label = []
    for img_ct in range(img_num):
        ret_label.append(label_set[img_ct])
    return ret_img, ret_label


def create_hdf5_data():
    _start_time = time.time()
    if not os.path.exists("HDF5_MNIST_TRAIN.h5"):
        _img, _label = read_mnist_data("train")
        _f = h5py.File("HDF5_MNIST_TRAIN.h5", 'w')
        _f["img"] = _img
        _f["label"] = _label
        _f.close()
    _end_time = time.time()
    print("used time for creating train data :", _end_time - _start_time)
    _start_time = time.time()
    if not os.path.exists("HDF5_MNIST_TEST.h5"):
        _img, _label = read_mnist_data("test")
        _f = h5py.File("HDF5_MNIST_TEST.h5", 'w')
        _f["img"] = _img
        _f["label"] = _label
        _f.close()
    _end_time = time.time()
    print("used time for creating test data:", _end_time - _start_time)


if __name__=='__main__':
    create_hdf5_data()