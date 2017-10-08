# File: VGG_pre_trained.py
# Author: Qian Ge <geqian1001@gmail.com>
import os

import numpy as np
import tensorflow as tf

from tensorcv.dataflow.image import *

import VGG
import config as config_path

def resize_tensor_image_with_smallest_side(image, small_size):
    """
    Resize image tensor with smallest side = small_size and
    keep the original aspect ratio.

    Args:
        image (tf.tensor): 4-D Tensor of shape [batch, height, width, channels] 
            or 3-D Tensor of shape [height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.

    Returns:
        Image tensor with the original aspect ratio and 
        smallest side = small_size .
        If images was 4-D, a 4-D float Tensor of shape 
        [batch, new_height, new_width, channels]. 
        If images was 3-D, a 3-D float Tensor of shape 
        [new_height, new_width, channels].       
    """
    im_shape = tf.shape(image)
    shape_dim = image.get_shape()
    if len(shape_dim) <= 3:
        height = tf.cast(im_shape[0], tf.float32)
        width = tf.cast(im_shape[1], tf.float32)
    else:
        height = tf.cast(im_shape[1], tf.float32)
        width = tf.cast(im_shape[2], tf.float32)

    height_smaller_than_width = tf.less_equal(height, width)

    new_shorter_edge = tf.constant(small_size, tf.float32)
    new_height, new_width = tf.cond(
    height_smaller_than_width,
    lambda: (new_shorter_edge, (width/height)*new_shorter_edge),
    lambda: ((height/width)*new_shorter_edge, new_shorter_edge))

    return tf.image.resize_images(tf.cast(image, tf.float32), 
        [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])

def get_word_list(file_path):
    word_dict = {}
    word_file = open(os.path.join(file_path), 'r')
    lines = word_file.read().split('\n')
    for i, line in enumerate(lines):
        label, word = line.split(' ', 1)
        word_dict[i] = word
    return word_dict

if __name__ == '__main__':

    # Setup inputs
    keep_prob = 1.0
    image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, None, None, 3])

    # Resize image with smallest side = 224
    input_im = resize_tensor_image_with_smallest_side(image, 224)

    # Create VGG-FCN model
    # Pre-trained parameters will be loaded if is_load = True
    VGG19 = VGG.VGG19_FCN(is_load = True, pre_train_path = config_path.vgg_dir)
    VGG19.create_model([input_im, keep_prob])

    # Top 5 predictions
    predict_op = tf.nn.top_k(tf.nn.softmax(VGG19.avg_output), 
                            k = 5, sorted = True)

    # Read image dataflow from a folder
    dataset_test = ImageFromFile('.JPEG', 
                                num_channel = 3,
                                data_dir = config_path.test_data_dir, 
                                shuffle = False)

    # Batch size has to be 1 if images have different size
    dataset_test.setup(epoch_val = 0, batch_size = 1)
    word_dict = get_word_list('./imageNetLabel.txt')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Pre-trained parameters are loaded by setting is_load = True 
        # Load pre-trained parameter
        # VGG19.load_pre_trained(sess, config_path.model_dir + 'vgg19.npy')

        # Test first 50 image in the folder
        for k in range(0, 50):
            batch_data = dataset_test.next_batch()
            result = sess.run(predict_op, feed_dict = {image: batch_data[0]})
            for val, ind in zip(result.values, result.indices):
                print(val)
                print(ind)
                print(word_dict[ind[0]])

    



 