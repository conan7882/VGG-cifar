#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_finetuning.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel

from vgg import VGG19_conv

class VGG19_Finetune(BaseModel):
    def __init__(self, num_class=1000,
                 num_channels=3,
                 im_height=None, im_width=None,
                 learning_rate=0.0001,
                 is_load=False,
                 pre_train_path=None,
                 is_rescale=False,
                 trainable_conv_12=False,
                 trainable_conv_3up=False,
                 trainable_fc=False):

        self._lr = learning_rate
        self.nchannel = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.nclass = num_class
        self._is_rescale = is_rescale
        self._train_low = trainable_conv_12
        self._train_high = trainable_conv_3up
        self._train_fc = trainable_fc

        self.layer = {}

        self._is_load = is_load
        if self._is_load and pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self.im_height, self.im_width, self.nchannel])

        self.label = tf.placeholder(tf.int64, [None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder([self.image, self.label])

    def _create_model(self):
        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        vgg_conv = VGG19_conv(num_class=self.nclass,
                              num_channels=self.nchannel,
                              is_load=self._is_load,
                              pre_train_path=self._pre_train_path ,
                              is_rescale=self._is_rescale,
                              trainable_conv_12=self._train_low,
                              trainable_conv_3up=self._train_high)

        vgg_conv.create_model([input_im, keep_prob])
        conv_out = vgg_conv.layer['pool5']

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([fc], trainable=self._train_fc):
            fc6 = fc(conv_out, 4096, 'fc6', nl=tf.nn.relu)
            dropout_fc6 = dropout(fc6, keep_prob, self.is_training)

            fc7 = fc(dropout_fc6, 4096, 'fc7', nl=tf.nn.relu)
            dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

            fc8 = fc(dropout_fc7, self.nclass, 'fc8')

            self.layer['fc6'] = fc6
            self.layer['fc7'] = fc7
            self.layer['fc8'] = self.layer['output'] = fc8
            self.layer['class_prob'] = tf.nn.softmax(fc8, name='class_prob')
            self.layer['pre_prob'] = tf.reduce_max(self.layer['class_prob'], axis=-1, name='pre_prob')

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(beta1=0.9,
                                      learning_rate=self._lr)

    def _get_loss(self):
        with tf.name_scope('loss'):
            cross_entropy =\
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.label,
                    logits=self.layer['output'])
            cross_entropy_loss = tf.reduce_mean(
                cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_loss)
            return tf.add_n(tf.get_collection('losses'), name='result')

    def get_train_op(self):
        grads = self.get_grads()
        opt = self.get_optimizer()
        train_op = opt.apply_gradients(grads, name='train')
        # self._setup_summery()

        return train_op

    def _ex_setup_graph(self):
        with tf.name_scope('accuracy'):
            prediction = tf.argmax(self.layer['output'], axis=-1)
            correct_prediction = tf.equal(prediction, self.label)
            self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32), 
                        name = 'result')
            tf.summary.scalar('accuracy', self.accuracy, collections=['train'])
