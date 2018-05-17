#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互熵损失(softmax分类器)
"""
import numpy as np
from data_utils import load_CIFAR10
# from classifiers.linear_classifier import LinearSVM
# import matplotlib.pyplot as plt
from classifiers.softmax import softmax_loss_naive
# from gradient_check import grad_check_sparse
# import sys
# import time

# 载入CIFAR-10数据集
cifar10_dir = r'machine_learning_study/dataset/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 抽取训练集/交叉验证集/测试集
num_training = 49000
num_validation = 1000
num_test = 1000

# 取图
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# 预处理：把数据展成一列
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# 预处理：减去图像均值
# 先求出训练集的均值
mean_image = np.mean(X_train, axis=0)
"""
print(mean_image[:10])  # print a few of the elements
plt.figure(figsize=(4, 4))
plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))  # 可视化一下
"""

# 然后从训练集和测试集里面减去图像均值
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

# 咱们把bias那一列1都加上
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

print(X_train.shape, X_val.shape, X_test.shape)

# 具体的softmax分类器代码可以看 classifiers/softmax.py
# 同样实现一个非常易懂但是效率很低的softmax损失函数计算

# 随便初始化一个权重序列，然后计算损失函数
W = np.random.randn(10, 3073) * 0.0001
loss, grad = softmax_loss_naive(W, X_train, y_train, 0.0)

# 总共有10类，如果我们随机猜，损失应该是-log(0.1).
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))
