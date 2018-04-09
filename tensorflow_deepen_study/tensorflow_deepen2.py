#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensorflow学习，helloworld
"""
import numpy as np

# 定义一个一维的数组
vector = np.array([1, 2, 3])

print(vector.shape)

# <class 'numpy.ndarray'>
print(type(vector))

# 维度
print(vector.ndim)

# 创建一个一维数组
one = np.arange(12)

print(one)

print(one.reshape(3, 4))
