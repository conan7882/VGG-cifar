#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys

sys.path.append('../')
from src.dataflow.images import Image 

def read_image(im_name, n_channel, data_dir='', batch_size=1):
    image_data = Image(
        im_name=im_name,
        data_dir=data_dir,
        n_channel=n_channel,
        shuffle=False,
        batch_dict_name=['image'],
        pf_list=None)
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