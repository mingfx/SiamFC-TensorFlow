#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Save the paths of crops from the ImageNet VID 2015 dataset in pickle format"""

#核心功能：将Data Curation处理后的训练数据存储为.pickle格式（train_imdb.pickle 和 validation_imdb.pickle）

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import os.path as osp
import pickle   # 导入pickle模块，将VID数据存储为pickle格式
import sys

import numpy as np
import tensorflow as tf

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))    #  添加搜索路径

from utils.misc_utils import sort_nicely


class Config:  # 参数配置
  ### Dataset
  # directory where curated dataset is stored
  dataset_dir = 'data/ILSVRC2015-VID-Curation'  # 之前curated数据测存储路径，可使用绝对路径或相对路径
  save_dir = 'data/'   # imdb文件存储路径

  # percentage of all videos for validation
  validation_ratio = 0.1   # validation所占的数据比例


class DataIter:
  """Container for dataset of one iteration"""
  pass


class Dataset:  # 数据集类
  def __init__(self, config):  # 初始化参数设置
    self.config = config

  def _get_unique_trackids(self, video_dir):  # 根据video_dir来获取特定的跟踪目标
    """Get unique trackids within video_dir"""
    x_image_paths = glob.glob(video_dir + '/*.crop.x.jpg')   # 取得video_dir下所有crop图片存储的文件名
    trackids = [os.path.basename(path).split('.')[1] for path in x_image_paths]  # 根据文件名获取对应trackid
    unique_trackids = set(trackids)  # 因为每个视频snippet都可能有很多段小视频，每个小视频里面又会有多个跟踪的目标
    return unique_trackids

  def dataset_iterator(self, video_dirs):  # 将所有的videos进行处理
    video_num = len(video_dirs)  # videos的数量
    iter_size = 150  # 每次循环处理的视频个数
    iter_num = int(np.ceil(video_num / float(iter_size)))#循环次数
    for iter_ in range(iter_num):
      iter_start = iter_ * iter_size
      iter_videos = video_dirs[iter_start: iter_start + iter_size]  # 每次处理的videos数量为150

      data_iter = DataIter()
      num_videos = len(iter_videos)
      instance_videos = []
      for index in range(num_videos):
        print('Processing {}/{}...'.format(iter_start + index, video_num))
        video_dir = iter_videos[index]   # 单个视频路径
        trackids = self._get_unique_trackids(video_dir)  # 该视频对应到有多少个target trackid

        for trackid in trackids:
          instance_image_paths = glob.glob(video_dir + '/*' + trackid + '.crop.x.jpg')  #trackid所有图片的路径

          # sort image paths by frame number
          instance_image_paths = sort_nicely(instance_image_paths)   # 根据image number排序

          # get image absolute path   # 获得排序好的图片计算对应的图片的绝对路径
          instance_image_paths = [os.path.abspath(p) for p in instance_image_paths]
          instance_videos.append(instance_image_paths)
      data_iter.num_videos = len(instance_videos)  # 150个视频里面trackid的数量，>=150
      data_iter.instance_videos = instance_videos
      yield data_iter  # yield返回一个生成器

  def get_all_video_dirs(self):  # 获取VID数据集中所有videos的路径
    ann_dir = os.path.join(self.config.dataset_dir, 'Data', 'VID')
    all_video_dirs = []
    # 根据之前预处理的训练数据存储的文件结构进行解析获取所有trackid的路径
    # We have already combined all training and validation videos in ILSVRC2015 and put them in the `train` directory.
    # The file structure is like:
    # train
    #    |- a
    #    |- b
    #    |_ c
    #       |- ILSVRC2015_train_00024001
    #       |- ILSVRC2015_train_00024002
    #       |_ ILSVRC2015_train_00024003
    #               |- 000045.00.crop.x.jpg
    #               |- 000046.00.crop.x.jpg
    #               |- ...
    train_dirs = os.listdir(os.path.join(ann_dir, 'train'))
    for dir_ in train_dirs:
      train_sub_dir = os.path.join(ann_dir, 'train', dir_)
      video_names = os.listdir(train_sub_dir)
      train_video_dirs = [os.path.join(train_sub_dir, name) for name in video_names]
      all_video_dirs = all_video_dirs + train_video_dirs

    return all_video_dirs


def main():
  # Get the data.
  config = Config()  # 加载参数设置
  dataset = Dataset(config)  # dataset实例
  all_video_dirs = dataset.get_all_video_dirs()  # 获取所有VID中videos的路径
  num_validation = int(len(all_video_dirs) * config.validation_ratio)  # validation数据数量

  ### validation 数据
  validation_dirs = all_video_dirs[:num_validation]
  validation_imdb = dict()
  validation_imdb['videos'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(validation_dirs)):
    validation_imdb['videos'] += data_iter.instance_videos
  validation_imdb['n_videos'] = len(validation_imdb['videos'])   # 视频数量
  validation_imdb['image_shape'] = (255, 255, 3)   # 图片大小

  ### train 数据
  train_dirs = all_video_dirs[num_validation:]
  train_imdb = dict()
  train_imdb['videos'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(train_dirs)):
    train_imdb['videos'] += data_iter.instance_videos
  train_imdb['n_videos'] = len(train_imdb['videos'])
  train_imdb['image_shape'] = (255, 255, 3)

  if not tf.gfile.IsDirectory(config.save_dir):   # imdb数据存储路径
    tf.logging.info('Creating training directory: %s', config.save_dir)
    tf.gfile.MakeDirs(config.save_dir)

  with open(os.path.join(config.save_dir, 'validation_imdb.pickle'), 'wb') as f:
    pickle.dump(validation_imdb, f)
  with open(os.path.join(config.save_dir, 'train_imdb.pickle'), 'wb') as f:
    pickle.dump(train_imdb, f)


if __name__ == '__main__':
  main()
