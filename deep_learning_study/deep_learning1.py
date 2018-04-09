#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从sklearn导入mnist数据集
"""
from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata(
    'MNIST Original', data_home='deep_learning_study/datasource')
print(mnist.get('data').shape)

print(np.unique(mnist.get('target')))
