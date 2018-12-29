#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from datasets.sampler import Sampler
from datasets.transforms import Compose, RandomGray, RandomCrop, CenterCrop, RandomStretch
from datasets.vid import VID
from utils.misc_utils import get


class DataLoader(object):
  def __init__(self, config, is_training):
    self.config = config
    self.is_training = is_training

    preprocess_name = get(config, 'preprocessing_name', None)#获取config中的“preprocessing”配置参数（color，gray，none。。）
    logging.info('preproces -- {}'.format(preprocess_name))

    if preprocess_name == 'siamese_fc_color':
      self.v_transform = None
      # TODO: use a single operation (tf.image.crop_and_resize) to achieve all transformations ?
      self.z_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8),
                                  CenterCrop((127, 127))])
      self.x_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8), ])
    elif preprocess_name == 'siamese_fc_gray':
      self.v_transform = RandomGray()
      self.z_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8),
                                  CenterCrop((127, 127))])
      self.x_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8), ])
    elif preprocess_name == 'None':
      self.v_transform = None
      self.z_transform = CenterCrop((127, 127))
      self.x_transform = CenterCrop((255, 255))
    else:
      raise ValueError('Preprocessing name {} was not recognized.'.format(preprocess_name))

    self.dataset_py = VID(config['input_imdb'], config['max_frame_dist'])#config中的imput_imdb为输入路径
    self.sampler = Sampler(self.dataset_py, shuffle=is_training)

  def build(self):
    self.build_dataset()
    self.build_iterator()

  '''
  一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 
  看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
  虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。
  看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。
  '''
  #构建generator来读取数据，用在from_generator方法里从而批量读取数据
  def build_dataset(self):
    def sample_generator():
      for video_id in self.sampler:#作用：随机取数据
        sample = self.dataset_py[video_id]#dataset_py视频数据
        yield sample

    def transform_fn(video):
      exemplar_file = tf.read_file(video[0])
      instance_file = tf.read_file(video[1])
      exemplar_image = tf.image.decode_jpeg(exemplar_file, channels=3, dct_method="INTEGER_ACCURATE")
      instance_image = tf.image.decode_jpeg(instance_file, channels=3, dct_method="INTEGER_ACCURATE")

      if self.v_transform is not None:
        video = tf.stack([exemplar_image, instance_image])
        video = self.v_transform(video)
        exemplar_image = video[0]
        instance_image = video[1]

      if self.z_transform is not None:
        exemplar_image = self.z_transform(exemplar_image)

      if self.x_transform is not None:
        instance_image = self.x_transform(instance_image)

      return exemplar_image, instance_image

    dataset = tf.data.Dataset.from_generator(sample_generator,
                                             output_types=(tf.string),
                                             output_shapes=(tf.TensorShape([2])))   #2维tensor
    dataset = dataset.map(transform_fn, num_parallel_calls=self.config['prefetch_threads'])
    dataset = dataset.prefetch(self.config['prefetch_capacity'])
    dataset = dataset.repeat()#feed数据重复的次数。1个epoch等于使用训练集中的全部样本训练一次；
    dataset = dataset.batch(self.config['batch_size'])#batch_size：每次训练在训练集中取batchsize个样本训练；
    self.dataset_tf = dataset

  def build_iterator(self):
    #使用make_one_shot_iterator()函数，构建一个iterator。
    self.iterator = self.dataset_tf.make_one_shot_iterator()

  def get_one_batch(self):
    return self.iterator.get_next()
