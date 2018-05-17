#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================
Restricted Boltzmann Machine features for digit classification
Restricted Boltzmann Machine，无监督学习的一种
思想：按照网络向前运算，当到达最后一个隐藏层的时候，用隐藏层作为输入
反向运算到第一层，将结果与当前的输入层对比，不断优化，达到一定的目的。
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

In order to learn good latent representations from a small dataset, we
artificially generate more labeled data by perturbing the training data with
linear shifts of 1 pixel in each direction.

This example shows how to build a classification pipeline with a BernoulliRBM
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the BernoulliRBM help improve the
classification accuracy.
比较原始像素值的逻辑回归。实例表明，通过bernoullirbm提取的特征有助于提高分类精度。
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


def shift(x, w):
    return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [[[0, 1, 0], [0, 0, 0],
                          [0, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 1],
                          [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]]]

    # numpy.apply_along_axis(func, axis, arr, *args, **kwargs)：
    # 必选参数：func,axis,arr。其中func是我们自定义的一个函数，函数func(arr)中的arr是一个数组，函数的主要功能就是对数组里的每一个元素进行变换，得到目标的结果。
    # 其中axis表示函数func对数组arr作用的轴。
    # 可选参数：*args, **kwargs。都是func()函数额外的参数。
    # 返回值：numpy.apply_along_axis()函数返回的是一个根据func()函数以及维度axis运算后得到的的数组.
    X = np.concatenate([X] + [
        np.apply_along_axis(shift, 1, X, vector)
        for vector in direction_vectors
    ])
    Y = np.concatenate([Y for _ in range(5)], axis=0)

    return X, Y


# Load Data
digits = datasets.load_digits()
X = np.asarray(digits.get('data'), 'float32')
X, Y = nudge_dataset(X, digits.get('target'))
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

# 这个例子展示了建立管道，先用bernoullirbm特征提取器，然后用线性模型的逻辑回归分类
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.（这里我们不进行交叉验证来节省时间。）
rbm.learning_rate = 0.06
# Number of iterations/sweeps over the training dataset to perform during
# training.
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time

# Number of binary hidden units.
rbm.n_components = 100

# 正则化系数，默认L2
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" %
      (metrics.classification_report(Y_test, classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" %
      (metrics.classification_report(Y_test,
                                     logistic_classifier.predict(X_test))))

###############################################################################
# Plotting

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(
        comp.reshape((8, 8)),
        cmap=plt.get_cmap('gray'),
        interpolation='nearest')

    plt.xticks(())
    plt.yticks(())

plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
