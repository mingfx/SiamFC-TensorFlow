#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Dataset Sampler"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Sampler(object):
  def __init__(self, data_source, shuffle=True):
    self.data_source = data_source
    self.shuffle = shuffle

  def __iter__(self):
    data_idxs = np.arange(len(self.data_source))#创建输入数据同样长度的数组
    if self.shuffle:
      np.random.shuffle(data_idxs)#打乱数组顺序

    for idx in data_idxs:
      yield idx#每次输出一个随机数字


if __name__ == '__main__':
  x = [1, 2, 3]
  sampler = Sampler(x, shuffle=True)
  p = 0
  for xx in sampler:
    print(x[xx])
    p += 1
    if p == 10: break
