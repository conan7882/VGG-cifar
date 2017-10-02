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


VGG_MEAN = [103.939, 116.779, 123.68]

# def get_predictConfig(FLAGS):
#     mat_name_list = ['level1Edge']
#     dataset_test = MatlabData('Level_1', shuffle = False,
#                                mat_name_list = mat_name_list,
#                                data_dir = FLAGS.test_data_dir)
#     prediction_list = PredictionImage(['prediction/label', 'prediction/probability'], 
#                                       ['test','test_pro'], 
#                                       merge_im = True)

#     return PridectConfig(
#                 dataflow = dataset_test,
#                 model = Model(FLAGS.input_channel, 
#                                 num_class = FLAGS.num_class),
#                 model_name = 'model-14070',
#                 model_dir = FLAGS.model_dir,    
#                 result_dir = FLAGS.result_dir,
#                 predictions = prediction_list,
#                 batch_size = FLAGS.batch_size)

if __name__ == '__main__':
    VGG19 = VGG.VGG19_FCN(num_class = 1000, 
                num_channels = 3, 
                im_height = 224, 
                im_width = 224)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, None, None, 3])
    input_im = image

    # input_im = tf.image.resize_images(tf.cast(image, tf.float32), [224, 224])

    VGG19.create_model([input_im, keep_prob])
    predict_op = tf.argmax(VGG19.output, dimension = -1)

    # dataset_val = ImageLabelFromFile('.JPEG', data_dir = config.valid_data_dir, 
    #                                 label_file_name = 'val_annotations.txt',
    #                                 num_channel = 3, label_dict = {},
    #                                 shuffle = False)
    dataset_val = ImageData('.JPEG', data_dir = config.valid_data_dir, 
                            shuffle = False)
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
        batch_data = dataset_val.next_batch()
        result = sess.run(predict_op, feed_dict = {keep_prob: 1, image: batch_data[0]})
        print(result.shape)
        print(result)

        batch_data = dataset_val.next_batch()
        result = sess.run(predict_op, feed_dict = {keep_prob: 1, image: batch_data[0]})
        print(result.shape)
        print(result)
        # print([word_dict[o_label_dict[label]] for label in batch_data[1]])

    



 