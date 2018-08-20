#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import numpy as np
import skimage.transform

sys.path.append('../')
from src.dataflow.images import Image
from src.dataflow.cifar import CIFAR 

def load_label_dict():
    label_dict = {}
    with open('../imageNetLabel.txt', 'r') as f:
        for idx, line in enumerate(f):
            names = line.rstrip()[10:]
            label_dict[idx] = names
    return label_dict

def read_image(im_name, n_channel, data_dir='', batch_size=1):

    def rescale_im(im):
        im = np.array(im)
        h, w = im.shape[0], im.shape[1]
        if h >= w:
            new_w = 224
            im = skimage.transform.resize(im, (int(h * new_w / w), 224),
                                          preserve_range=True)
        else:
            new_h = 224
            im = skimage.transform.resize(im, (224, int(w * new_h / h)),
                                          preserve_range=True)
        return im.astype('uint8')

    image_data = Image(
        im_name=im_name,
        data_dir=data_dir,
        n_channel=n_channel,
        shuffle=False,
        batch_dict_name=['image'],
        pf_list=rescale_im)
    image_data.setup(epoch_val=0, batch_size=batch_size)

    return image_data

def load_cifar(cifar_path, batch_size=64, substract_mean=True):
    # cifar_path = 'E:/Dataset/cifar/'
    # im_pair = [im[:,:,::-1,:] for im in im_pair]
    def preprocess(im):
        if np.random.random() > 0.5:
            im = im[:,::-1,:]
        # if np.random.random() > 0.5:
        #     im = im[::-1,:,:] 
        return im

    train_data = CIFAR(
        data_dir=cifar_path,
        shuffle=True,
        batch_dict_name=['image', 'label'],
        data_type='train',
        channel_mean=None,
        substract_mean=substract_mean,
        pf=preprocess)
    train_data.setup(epoch_val=0, batch_size=batch_size)

    valid_data = CIFAR(
        data_dir=cifar_path,
        shuffle=False,
        batch_dict_name=['image', 'label'],
        data_type='valid',
        channel_mean=train_data.channel_mean,
        substract_mean=substract_mean)
    valid_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data, valid_data

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # image_data = read_image(im_name='.png', n_channel=3, data_dir='../fig/')
    image_data, _ = load_cifar('E:/Dataset/cifar/')
    batch_data = image_data.next_batch_dict()
    plt.figure()
    plt.imshow(np.squeeze(batch_data['image'][0]))
    plt.show()
    