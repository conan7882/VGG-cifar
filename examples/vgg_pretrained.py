#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_pretrained.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')
import loader as loader
from src.nets.vgg import VGG19

VGG_PATH = '/Users/gq/workspace/Dataset/pretrained/vgg19.npy'
DATA_PATH = '../fig/'
IM_CHANNEL = 3


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vgg_path', type=str, default=VGG_PATH,
                        help='Path of pretrain VGG19 model')
    # parser.add_argument('--n_channel', type=int, default=3,
    #                     help='Number of channels of input images')
    parser.add_argument('--im_image', type=str, default='.png',
                        help='Part of image image name')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                        help='Path to put test image data')
    
    return parser.parse_args()

def test_pre_trained():
    FLAGS = get_args()
    image_data = loader.read_image(
        im_name=FLAGS.im_image, n_channel=IM_CHANNEL,
        data_dir=FLAGS.data_path, batch_size=1)

    test_model = VGG19(
        n_channel=IM_CHANNEL, n_class=1000, pre_trained_path=FLAGS.vgg_path)
    test_model.create_test_model()


if __name__ == "__main__":
    test_pre_trained()

