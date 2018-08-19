#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_imagenet_pretrained.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import tensorflow as tf

from tensorcv.dataflow.image import ImageFromFile

import vgg
import config as config_path


def get_word_list(file_path):
    word_dict = {}
    word_file = open(os.path.join(file_path), 'r')
    lines = word_file.read().split('\n')
    for i, line in enumerate(lines):
        label, word = line.split(' ', 1)
        word_dict[i] = word
    return word_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='.jpg', type=str,
                        help='image type for training and testing')

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()

    # Setup inputs
    keep_prob = 1.0
    image = tf.placeholder(tf.float32, name='image',
                           shape=[None, None, None, 3])

    # Create VGG-FCN model
    # Pre-trained parameters will be loaded if is_load = True
    VGG19 = VGG.VGG19_FCN(is_load=True,
                          pre_train_path=config_path.vgg_dir,
                          is_rescale=True)
    VGG19.create_model([image, keep_prob])

    # Top 5 predictions
    predict_op = tf.nn.top_k(tf.nn.softmax(VGG19.avg_output),
                             k=5, sorted=True)

    # Read image dataflow from a folder
    dataset_test = ImageFromFile(FLAGS.type,
                                 num_channel=3,
                                 data_dir=config_path.test_data_dir,
                                 shuffle=False)

    # Batch size has to be 1 if images have different size
    dataset_test.setup(epoch_val=0, batch_size=1)
    word_dict = get_word_list('./imageNetLabel.txt')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Test first 50 image in the folder
        for k in range(0, 50):
            if dataset_test.epochs_completed < 1:
                batch_data = dataset_test.next_batch()
                result = sess.run(predict_op, feed_dict={image: batch_data[0]})
                for val, ind in zip(result.values, result.indices):
                    print(val)
                    print(ind)
                    print(word_dict[ind[0]])
