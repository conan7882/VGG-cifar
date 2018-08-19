#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import numpy as np
import scipy.misc
import tensorflow as tf

import sys
sys.path.append('../')
from lib.nets.vgg_new import BaseVGG19
from lib.dataflow.cifar import CIFAR
from lib.helper.trainer import Trainer


IM_SIZE = 32
# vgg_path = '/Users/gq/workspace/Dataset/pretrained/vgg19.npy'
vgg_path = '/home/qge2/workspace/data/pretrain/vgg/vgg19.npy'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0005, type=float)
    # parser.add_argument('--dropout', default=0.5, type=float)
    # parser.add_argument('--wd', default=0, type=float)
    # parser.add_argument('--epoch', default=150, type=int)

    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--viz', action='store_true')

    return parser.parse_args()

def im_rescale(im, resize):
    im_shape = im.shape
    im = scipy.misc.imresize(im, (resize[0], resize[1], im_shape[-1]))
    return im

def im_scale(im):
    return im_rescale(im, [IM_SIZE, IM_SIZE])

if __name__ == '__main__':
    FLAGS = get_args()
    # file_path = '../fish.jpg'
    # im = np.array([im_scale(scipy.misc.imread(file_path, mode='RGB'))])
    # print(im.shape)

    train_data = CIFAR(
        data_dir='/home/qge2/workspace/data/dataset/cifar/',
        shuffle=True,
        batch_dict_name=['im', 'label'],
        data_type='train',
        substract_mean=True)
    train_data.setup(epoch_val=0, batch_size=128)

    test_data = CIFAR(
        data_dir='/home/qge2/workspace/data/dataset/cifar/',
        shuffle=False,
        batch_dict_name=['im', 'label'],
        data_type='test',
        channel_mean=train_data.channel_mean,
        substract_mean=True)
    test_data.setup(epoch_val=0, batch_size=128)

    model = BaseVGG19(num_class=10,
                      num_channels=3,
                      im_height=IM_SIZE,
                      im_width=IM_SIZE,
                      is_load=False,
                      pre_train_path=vgg_path,
                      trainable=True)

    trainer = Trainer(model, train_data, FLAGS.lr)
    # train_op = model.get_train_op()
    # loss_op = model.get_loss()
    # acc_op = model.get_accuracy()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 10000):
            trainer.train_epoch(sess, keep_prob=0.4)
            trainer.valid_epoch(sess, test_data)
            # batch_data = train_data.next_batch_dict()
            # _, loss = sess.run(
            #     [train_op, loss_op],
            #     feed_dict={model.image: batch_data['im'],
            #                model.keep_prob: 0.5,
            #                model.label: batch_data['label'],
            #                model.lr: FLAGS.lr})
            # print(loss)
        # top5 = sess.run(model.layers['pred5'], feed_dict={model.image: im, model.keep_prob: 1.0})
        # print(top5)

