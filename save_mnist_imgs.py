import h5py
import os
# import sys
import PIL.Image as Image
import numpy as np


def save_imgs(imgs, labels, parent_dir):
    for i in imgs.shape[0]:
        img = imgs[i].astype(np.uint8)
        dir_path = os.path.join(parent_dir, '%s' % labels[i])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        Image.fromarray(img).save(os.path.join(dir_path, '%s.jpg' % i))


if __name__=='__main__':
    if os.path.exists("./HDF5_MNIST_TRAIN.h5"):
        f = h5py.File("./HDF5_MNIST_TRAIN.h5", 'r')
        save_imgs(f["img"][:], f["label"][:], 'train')
        f.close()
    else:
        print('Train Img Save Failed!')
    if os.path.exists("./HDF5_MNIST_TEST.h5"):
        f = h5py.File("./HDF5_MNIST_TEST.h5", 'r')
        save_imgs(f["img"][:], f["label"][:], 'test')
        f.close()
    else:
        print("Test Img Save Failed!")
