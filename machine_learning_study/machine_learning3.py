#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hinge loss/支持向量机损失
"""
import numpy as np
from data_utils import load_CIFAR10
from classifiers.linear_classifier import LinearSVM
import matplotlib.pyplot as plt
# from classifiers.linear_svm import svm_loss_vectorized
# from gradient_check import grad_check_sparse
# import sys
import time

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
"""
# 评估一下用for循环完成的svm_loss_naive的效果和效率:

# 产出SVM的初始权重
W = np.random.randn(10, 3073) * 0.0001
loss, grad = svm_loss_vectorized(W, X_train, y_train, 0.00001)
print('loss: %f' % (loss, ))


def f(w):
    # 梯度检查，要检查数值梯度和解析梯度是否一致，因为解析梯度计算快，但是容易出错
    return svm_loss_vectorized(w, X_train, y_train, 0.0)[0]


grad_check_sparse(f, W, grad, 10)
"""

# 实现随机梯度下降，再run一下

svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(
    X_train,
    y_train,
    learning_rate=1e-7,
    reg=5e4,
    num_iters=1500,
    verbose=True)
toc = time.time()
print('That took %fs' % (toc - tic))

# 我们画出来随着迭代次数增多，损失的变化状况
plt.figure()
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')

# 对样本进行预测并计算准确度
y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))

# 可视化每个类对应的权重
# 需要多说一句的是，因为初始值和学习率等的不同，你看到的结果可能会有一些差别
w = svm.W[:, :-1]  # strip out the bias
w = w.reshape(10, 32, 32, 3)
w_min, w_max = np.min(w), np.max(w)
classes = [
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck'
]

plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

plt.show()
