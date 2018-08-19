#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_module.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

import src.models.layers as L


def sub_rgb2bgr_mean(inputs):
    with tf.name_scope('sub_mean'):
        red, green, blue = tf.split(axis=3,
                                    num_or_size_splits=3,
                                    value=inputs)

        vgg_mean = [103.939, 116.779, 123.68]

        input_bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])
        return input_bgr

def vgg19_conv(layer_dict, inputs=None, pretrained_dict=None,
               bn=False, init_w=None, trainable=True, is_training=True, wd=0):

    if inputs is not None:
        layer_dict['cur_input'] = inputs
    
    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                    bn=bn, nl=tf.nn.relu, init_w=init_w, trainable=trainable,
                    is_training=is_training, wd=wd, add_summary=False):

        L.conv(filter_size=3, out_dim=64, name='conv1_1')
        L.conv(filter_size=3, out_dim=64, name='conv1_2')
        L.max_pool(layer_dict, name='pool1')

        L.conv(filter_size=3, out_dim=128, name='conv2_1')
        L.conv(filter_size=3, out_dim=128, name='conv2_2')
        L.max_pool(layer_dict, name='pool2')

        L.conv(filter_size=3, out_dim=256, name='conv3_1')
        L.conv(filter_size=3, out_dim=256, name='conv3_2')
        L.conv(filter_size=3, out_dim=256, name='conv3_3')
        L.conv(filter_size=3, out_dim=256, name='conv3_4')
        L.max_pool(layer_dict, name='pool3')

        L.conv(filter_size=3, out_dim=512, name='conv4_1')
        L.conv(filter_size=3, out_dim=512, name='conv4_2')
        L.conv(filter_size=3, out_dim=512, name='conv4_3')
        L.conv(filter_size=3, out_dim=512, name='conv4_4')
        L.max_pool(layer_dict, name='pool4')

        L.conv(filter_size=3, out_dim=512, name='conv5_1')
        L.conv(filter_size=3, out_dim=512, name='conv5_2')
        L.conv(filter_size=3, out_dim=512, name='conv5_3')
        L.conv(filter_size=3, out_dim=512, name='conv5_4')
        L.max_pool(layer_dict, name='pool5')

        return layer_dict['cur_input']

def vgg16_conv(layer_dict, inputs=None, pretrained_dict=None,
               bn=False, init_w=None, trainable=True, is_training=True, wd=0):

    if inputs is not None:
        layer_dict['cur_input'] = inputs

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                    bn=bn, nl=tf.nn.relu, init_w=init_w, trainable=trainable,
                    is_training=is_training, wd=wd, add_summary=False):

        L.conv(filter_size=3, out_dim=64, name='conv1_1')
        L.conv(filter_size=3, out_dim=64, name='conv1_2')
        L.max_pool(layer_dict, name='pool1')

        L.conv(filter_size=3, out_dim=128, name='conv2_1')
        L.conv(filter_size=3, out_dim=128, name='conv2_2')
        L.max_pool(layer_dict, name='pool2')

        L.conv(filter_size=3, out_dim=256, name='conv3_1')
        L.conv(filter_size=3, out_dim=256, name='conv3_2')
        L.conv(filter_size=3, out_dim=256, name='conv3_3')
        L.max_pool(layer_dict, name='pool3')

        L.conv(filter_size=3, out_dim=512, name='conv4_1')
        L.conv(filter_size=3, out_dim=512, name='conv4_2')
        L.conv(filter_size=3, out_dim=512, name='conv4_3')
        L.max_pool(layer_dict, name='pool4')

        L.conv(filter_size=3, out_dim=512, name='conv5_1')
        L.conv(filter_size=3, out_dim=512, name='conv5_2')
        L.conv(filter_size=3, out_dim=512, name='conv5_3')
        L.max_pool(layer_dict, name='pool5')

        return layer_dict['cur_input']

def vgg_fc(layer_dict, n_class, keep_prob, inputs=None, pretrained_dict=None,
           bn=False, init_w=None, trainable=True, is_training=True, wd=0):

    if inputs is not None:
        layer_dict['cur_input'] = inputs

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                    bn=bn, init_w=init_w, trainable=trainable,
                    is_training=is_training, wd=wd, add_summary=False,
                    padding='VALID'):

        L.conv(filter_size=7, out_dim=4096, nl=tf.nn.relu, name='fc6')
        L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

        L.conv(filter_size=1, out_dim=4096, nl=tf.nn.relu, name='fc7')
        L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

        L.conv(filter_size=1, out_dim=n_class, name='fc8')

    return layer_dict['cur_input']

    def _creat_fc(self, inputs, keep_prob):
        self.cur_input = inputs
        data_dict = self.data_dict
        def fc_layer(out_dim, name, keep_prob):
            init_w = tf.keras.initializers.he_normal()
            # init_w = None
            self.layers[name] = fc(self.cur_input,
                                   out_dim,
                                   name,
                                   nl=tf.nn.relu,
                                   init_w=init_w)
            self.layers['dropout_{}'.format(name)] = dropout(
                self.layers[name], keep_prob, self.is_training)
            self.cur_input = self.layers['dropout_{}'.format(name)]

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([fc], trainable=self._trainable, data_dict=data_dict):
            fc_layer(4096, 'fc6', keep_prob)
            fc_layer(4096, 'fc7', keep_prob)

            init_w = tf.keras.initializers.he_normal()
            # init_w = None
            self.layers['fc8'] = fc(
                self.cur_input, self.num_class, 'fc8', init_w=init_w)

        return self.layers['fc8']

