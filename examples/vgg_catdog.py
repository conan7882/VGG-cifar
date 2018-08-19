#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_finetuning_ex.py
# Author: Qian Ge <geqian1001@gmail.com>

from collections import namedtuple
import argparse
import os

import tensorflow as tf

from tensorcv.train.config import TrainConfig
from tensorcv.predicts.config import PridectConfig
from tensorcv.predicts.predictions import *
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.callbacks import *
from tensorcv.dataflow.image import ImageFromFile, ImageLabelFromFolder

import setup_env
# from dataflow.dataset import ImageLabelFromFolder, new_ImageFromFile, separate_data
from models.vgg_catvsdog import VGG19_CatDog

TRAIN_DIR = '/home/qge2/workspace/data/dataset/dogvscat/train/'
VALID_DIR = '/home/qge2/workspace/data/dataset/dogvscat/val/'
TEST_DIR = '/home/qge2/workspace/data/dataset/dogvscat/test/'
VGG_PATH = '/home/qge2/workspace/data/pretrain/vgg/vgg19.npy'
SAVE_DIR = '/home/qge2/workspace/data/dogcat/'
configpath = namedtuple('CONFIG_PATH', ['summary_dir', 'checkpoint_dir', 'model_dir', 'result_dir'])
config_path = configpath(summary_dir=SAVE_DIR, checkpoint_dir=SAVE_DIR, model_dir=SAVE_DIR, result_dir=SAVE_DIR)


def display_data(dataflow, data_name):
    try:
        print('[{}] num of samples {}, num of classes {}'.\
            format(data_name, dataflow.size(), len(dataflow.label_dict)))
    except AttributeError:
        print('[{}] num of samples {}'.\
            format(data_name, dataflow.size()))

def config_train(FLAGS):
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    train_data = ImageLabelFromFolder('.jpg', data_dir=TRAIN_DIR,
                                      num_channel=3, resize=224)

    val_data = ImageLabelFromFolder('.jpg', data_dir=VALID_DIR,
                                    num_channel=3, resize=224)

    display_data(train_data, 'training data')
    display_data(val_data, 'validation data')

    n_class = len(train_data.label_dict)

    model = VGG19_CatDog(num_class=n_class,
                         num_channels=3,
                         learning_rate=FLAGS.lr,
                         is_load=True,
                         pre_train_path=VGG_PATH,
                         is_rescale=False,
                         im_height=224, im_width=224,
                         trainable_conv_12=False,
                         trainable_conv_3up=False,
                         trainable_fc=True,
                         drop_out=FLAGS.dropout)

    inference_list_validation = InferScalars(['accuracy/result', 'loss/result', 'loss/cross_entropy'],
                                             ['test_accuracy', 'test_loss', 'test_entropy'])

    training_callbacks = [
        ModelSaver(periodic=300),
        TrainSummary(key='train', periodic=10),
        FeedInferenceBatch(val_data, batch_count=20, periodic=10,
                           inferencers=inference_list_validation),
        CheckScalar(['accuracy/result', 'loss/result'], periodic=10)]

    return TrainConfig(
        dataflow=train_data, model=model,
        callbacks=training_callbacks,
        batch_size=32, max_epoch=25,
        monitors=TFSummaryWriter(),
        summary_periodic=10,
        is_load=False,
        model_name='bk6/model-5700',
        default_dirs=config_path)


# def config_predict():
#     # test_data = ImageLabelFromCSVFile('.jpg', data_dir=DATA_DIR, start_line=1,
#     #                                   label_file_name='../labels.csv',
#     #                                   num_channel=3, resize=224)
#     test_data = new_ImageFromFile('.jpg', data_dir=TEST_DIR, 
#                               num_channel=3,
#                               shuffle=False,
#                               resize=224)

#     pridection_list = [
#     # PredictionMeanScalar(prediction_scalar_tensors='accuracy/result', 
#     #                                         print_prefix='test_accuracy'),
#                     # PredictionScalar('pre_prob', 'out'),
#                     PredictionMat('class_prob','class_prob')]

#     n_class = 120
#     vgg_net = VGG19_Finetune(num_class=n_class,
#                              num_channels=3,
#                              is_load=False,
#                              is_rescale=False,
#                              im_height=224, im_width=224,
#                              trainable_conv_12=False,
#                              trainable_conv_3up=False,
#                              trainable_fc=False)

#     return PridectConfig(
#                  dataflow=test_data, model=vgg_net,
#                  model_dir=SAVE_DIR, model_name='bk2/model-3300',
#                  predictions=pridection_list,
#                  batch_size=128,
#                  default_dirs=config_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict', action='store_true',
                        help='Run prediction')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')

    parser.add_argument('--lr', default=1e-6, type=float,
                        help='learning rate of fine tuning')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='drop out keep probability')

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.train:
        config = config_train(FLAGS)
        SimpleFeedTrainer(config).train()
    # if FLAGS.predict:
    #     config = config_predict()
    #     SimpleFeedPredictor(config).run_predict()

# if __name__ == '__main__':
#     keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#     image = tf.placeholder(tf.float32, name='image',
#                            shape=[None, None, None, 3])

#     train_data = ImageLabelFromCSVFile('.jpg', data_dir=DATA_DIR, start_line=1,
#                               label_file_name='../labels.csv',
#                               num_channel=3)
#     n_class = len(train_data.label_dict)

#     vgg_net = VGG19(num_class=1000,
#                     num_channels=n_class,
#                     learning_rate=0.0001,
#                     is_load=True,
#                     pre_train_path=VGG_PATH,
#                     is_rescale=True,
#                     trainable_conv_12=False,
#                     trainable_conv_3up=True,
#                     trainable_fc=True)

#     vgg_net.create_model([image, keep_prob])
#     vgg_net.setup_graph()

#     train_op = vgg_net.get_train_op()

#     writer = tf.summary.FileWriter(SAVE_DIR)
#     saver = tf.train.Saver()

#     trigger_step = 10
#     faster_trigger_step = 10
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         # saver.restore(sess, LOAD_DIR + '-50000')

#         writer.add_graph(sess.graph)
#         batch_data = train_data.next_batch()

#         step_cnt = 1
#         sum_cls_cost = 0
#         sum_reg_cost = 0
#         while train_data.epochs_completed < 10:
#             batch_data = train_data.next_batch()
#             im = batch_data[0]
#             label = batch_data[1]
#             re = sess.run([train_op],
#                           feed_dict={image: im, keep_prob: 0.5, })
#             print('step: {}, cls_cost: {}, reg_cost: {}, cost: {}'.
#                   format(step_cnt, re[1], re[2], re[1] + re[2]))
