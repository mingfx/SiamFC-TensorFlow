#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

# 核心功能：根据SiamsesFC论文中的Curation方式进行图片预处理。

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import xml.etree.ElementTree as ET   # 解析xml文件的模块
from glob import glob
from multiprocessing.pool import ThreadPool  # 多进程加速

import cv2
from cv2 import imread, imwrite  # 使用 opencv 读写图像

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)  # 添加搜索路径

from utils.infer_utils import get_crops, Rectangle, convert_bbox_format   # 自己写的小函数
from utils.misc_utils import mkdir_p


def get_track_save_directory(save_dir, split, subdir, video):  # 存储路径的一个简单映射函数，简化存储目录
  subdir_map = {'ILSVRC2015_VID_train_0000': 'a',   # 将train映射到（a,b,c,d），val映射到（e）
                'ILSVRC2015_VID_train_0001': 'b',
                'ILSVRC2015_VID_train_0002': 'c',
                'ILSVRC2015_VID_train_0003': 'd',
                '': 'e'}
  return osp.join(save_dir, 'Data', 'VID', split, subdir_map[subdir], video)


def process_split(root_dir, save_dir, split, subdir='', ):   # 数据处理核心函数，spilt就是['val','train']中的值
  data_dir = osp.join(root_dir, 'Data', 'VID', split)   # 待处理图片数据路径
  anno_dir = osp.join(root_dir, 'Annotations', 'VID', split, subdir)   # 待处理图片的注释数据路径
  #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
  video_names = os.listdir(anno_dir)   # train和val数据集所有的video names，因为我们只有这两个数据集的annotations

  for idx, video in enumerate(video_names):
    print('{split}-{subdir} ({idx}/{total}): Processing {video}...'.format(split=split, subdir=subdir,
                                                                           idx=idx, total=len(video_names),
                                                                           video=video))
    video_path = osp.join(anno_dir, video)#
    #glob模块的主要方法就是glob,该方法返回所有匹配的文件路径列表（list）；
    xml_files = glob(osp.join(video_path, '*.xml'))   # 获得当前video的所有图片对应的.xml文件

    for xml in xml_files:
      tree = ET.parse(xml)  # 使用ET处理单个图片(.jpeg && .xml)
      root = tree.getroot()

      folder = root.find('folder').text   # 解析.xml文件的folder
      filename = root.find('filename').text  # 解析.xml文件中的filename

      # Read image
      img_file = osp.join(data_dir, folder, filename + '.JPEG')  # 将.xml文件名对应到相同的图片文件名
      img = None

      # Get all object bounding boxes
      bboxs = []
      for object in root.iter('object'):   # 找到所有的objects
        bbox = object.find('bndbox')    # 找到 box项，.xml文件中名称是bndbox
        xmax = float(bbox.find('xmax').text)  # 找到对应的 xmax并转换为float类型的box数据，以下类似
        xmin = float(bbox.find('xmin').text)
        ymax = float(bbox.find('ymax').text)
        ymin = float(bbox.find('ymin').text)
        width = xmax - xmin + 1    # 计算width 和 height
        height = ymax - ymin + 1
        bboxs.append([xmin, ymin, width, height])   # 返回的box的形式是[xmin,ymin,wedth,height]

      for idx, object in enumerate(root.iter('object')):
        id = object.find('trackid').text  # 获取object的trackid,因为同一个video中可能存在多个需要跟踪的目标，加以区分
        class_name = object.find('name').text  # 所属类别名称（VID中的30个大类）

        track_save_dir = get_track_save_directory(save_dir, 'train', subdir, video)   # 获取存储路径
        mkdir_p(track_save_dir)
        savename = osp.join(track_save_dir, '{}.{:02d}.crop.x.jpg'.format(filename, int(id)))
        if osp.isfile(savename): continue  # skip existing images  # 文件存在的情况下无需存储

        if img is None:
          img = imread(img_file)

        # Get crop
        target_box = convert_bbox_format(Rectangle(*bboxs[idx]), 'center-based')  # box格式转换
        crop, _ = get_crops(img, target_box,
                            size_z=127, size_x=255,
                            context_amount=0.5, )

        imwrite(savename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])   # 图片存储，质量为90


if __name__ == '__main__':
  vid_dir = osp.join(ROOT_DIR, 'data/ILSVRC2015')   # VID原始数据集存储的根目录，可以跟改为自己存储数据的绝对路径

  # Or, you could save the actual curated data to a disk with sufficient space
  # then create a soft link in `data/ILSVRC2015-VID-Curation`
  save_dir = 'data/ILSVRC2015-VID-Curation'  # 处理之后的数据存储的位置，可以修改为绝对路径

  pool = ThreadPool(processes=5)   # 开启5个线程加快处理速度

  one_work = lambda a, b: process_split(vid_dir, save_dir, a, b)  # 执行函数，下面调用的时候每个会在每个线程中执行

  results = []
  results.append(pool.apply_async(one_work, ['val', '']))  # 非阻塞方式，线程1调用one_work处理val数据集，下类似
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0000']))
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0001']))
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0002']))
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0003']))
  ans = [res.get() for res in results]    # 获取子线程结果，作用就是阻塞主线程，等待子线程执行完成
