#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import numpy as np
import skimage.transform

sys.path.append('../')
from src.dataflow.images import Image 

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

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    image_data = read_image(im_name='.png', n_channel=3, data_dir='../fig/')
    batch_data = image_data.next_batch_dict()
    plt.figure()
    plt.imshow(np.squeeze(batch_data['image'][0]))
    plt.show()