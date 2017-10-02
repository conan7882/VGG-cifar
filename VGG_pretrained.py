# File: VGG.py
# Author: Qian Ge <geqian1001@gmail.com>
import os

import numpy as np
import tensorflow as tf

from tensorcv.models.layers import *
from tensorcv.dataflow.image import *
from tensorcv.models.base import BaseModel

import VGG
import config

def resize_image_with_smallest_224(image):
    im_shape = tf.shape(image)
    shape_dim = image.get_shape()
    if len(shape_dim) <= 3:
        height = tf.cast(im_shape[0], tf.float32)
        width = tf.cast(im_shape[1], tf.float32)
    else:
        height = tf.cast(im_shape[1], tf.float32)
        width = tf.cast(im_shape[2], tf.float32)

    height_smaller_than_width = tf.less_equal(height, width)

    new_shorter_edge = tf.constant(224.0, tf.float32)
    new_height, new_width = tf.cond(
    height_smaller_than_width,
    lambda: (new_shorter_edge, (width/height)*new_shorter_edge),
    lambda: ((height/width)*new_shorter_edge, new_shorter_edge))


    return tf.image.resize_images(tf.cast(image, tf.float32), 
        [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])

if __name__ == '__main__':

    keep_prob = 1.0
    image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, None, None, 3])
    input_im = resize_image_with_smallest_224(image)


    # Create model
    VGG19 = VGG.VGG19_FCN(num_class = 1000)
    VGG19.create_model([input_im, keep_prob])
    # predict_op = tf.argmax(VGG19.avg_output, dimension = -1)
    predict_op = tf.nn.top_k(tf.nn.softmax(VGG19.avg_output), k = 5, sorted = True)

    dataset_val = ImageFromFile('.JPEG', 
                                num_channel = 3,
                                data_dir = config.valid_data_dir, 
                                shuffle = False)

    # batch size has to be 1 if images have different size
    dataset_val.setup(epoch_val = 0, batch_size = 1)
    # o_label_dict = dataset_val.label_dict_reverse

    # word_dict = {}
    # word_file = open(os.path.join('D:\\Qian\\GitHub\\workspace\\dataset\\tiny-imagenet-200\\tiny-imagenet-200\\', 
    #                                 'words.txt'), 'r')
    # lines = word_file.read().split('\n')
    # for line in lines:
    #     label, word = line.split('\t')
    #     word_dict[label] = word

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        VGG19.load_pre_trained(sess, config.model_dir + 'vgg19.npy')

        for k in range(0, 10):
            batch_data = dataset_val.next_batch()
            result = sess.run(predict_op, feed_dict = {image: batch_data[0]})
            print(result.values)
            print(result.indices)


        # print([word_dict[o_label_dict[label]] for label in batch_data[1]])

    



 