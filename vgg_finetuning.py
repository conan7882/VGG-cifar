#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_finetuning.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from dataset import ImageLabelFromCSVFile
from vgg import VGG19


DATA_DIR = '/Users/gq/workspace/Dataset/kaggle/dog_bleed/train/'
VGG_PATH = '/home/qge2/workspace/data/pretrain/vgg/vgg19.npy'
if __name__ == '__main__':
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	image = tf.placeholder(tf.float32, name='image',
                           shape=[None, None, None, 3])

    train_data = ImageLabelFromCSVFile('.jpg', data_dir=DATA_DIR, start_line=1,
                              label_file_name='../labels.csv',
                              num_channel=3)
    n_class = len(train_data.label_dict)

	vgg_net = VGG19(num_class=1000,
                 	num_channels=n_class,
                 	learning_rate=0.0001,
                 	is_load=True,
                 	pre_train_path=VGG_PATH,
                 	is_rescale=True,
                 	trainable_conv_12=False,
                 	trainable_conv_3up=True,
                 	trainable_fc=True)

	vgg_net.create_model([image, keep_prob])

	train_op = vgg_net.get_train_op()
