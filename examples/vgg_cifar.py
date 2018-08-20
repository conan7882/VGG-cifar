#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_cifar.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')
import loader as loader
from src.nets.vgg import VGG_CIFAR10
from src.helper.trainer import Trainer


# VGG_PATH = '/Users/gq/workspace/Dataset/pretrained/vgg19.npy'
# VGG_PATH = 'E:/GITHUB/workspace/CNN/pretrained/vgg19.npy'
# VGG_PATH = '/home/qge2/workspace/data/pretrain/vgg/vgg16.npy'
DATA_PATH = '/home/qge2/workspace/data/dataset/cifar/'
SAVE_PATH = '/home/qge2/workspace/data/out/vgg/cifar/'
# IM_CHANNEL = 3

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probability for dropout')
    parser.add_argument('--maxepoch', type=int, default=150,
                        help='Max number of epochs for training')

    # parser.add_argument('--vgg_path', type=str, default=VGG_PATH,
    #                     help='Path of pretrain VGG19 model')
    # parser.add_argument('--n_channel', type=int, default=3,
    #                     help='Number of channels of input images')
    # parser.add_argument('--im_image', type=str, default='.png',
    #                     help='Part of image image name')
    # parser.add_argument('--data_path', type=str, default=DATA_PATH,
    #                     help='Path to put test image data')
    
    return parser.parse_args()

def train():
    FLAGS = get_args()
    train_data, valid_data = loader.load_cifar(
        cifar_path=DATA_PATH, batch_size=FLAGS.bsize, substract_mean=True)

    train_model = VGG_CIFAR10(
        n_channel=3, n_class=10, pre_trained_path=None,
        bn=True, wd=5e-3, trainable=True, sub_vgg_mean=False)
    train_model.create_train_model()

    valid_model = VGG_CIFAR10(
        n_channel=3, n_class=10, bn=True, sub_vgg_mean=False)
    valid_model.create_test_model()

    trainer = Trainer(train_model, valid_model, train_data, init_lr=FLAGS.lr)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SAVE_PATH)
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for i in range(FLAGS.maxepoch):
            trainer.train_epoch(sess, keep_prob=FLAGS.keep_prob, summary_writer=writer)
            trainer.valid_epoch(sess, dataflow=valid_data, summary_writer=writer)

if __name__ == "__main__":
    train()