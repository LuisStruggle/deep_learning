#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监督学习算法：多层向前神经网络(Multilayer Feed-Forward Neural Network)
Backpropagation被使用在多层向前神经网络上
手写数字识别：

每个图片8x8的灰度图
识别数字：0,1,2,3,4,5,6,7,8,9
"""
from sklearn.datasets import load_digits
# MLP（MultiLayer Perceptions）多层传感器（多层神经元）
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
# import pylab as pl

# 获取数据集
digits = load_digits()
print(digits.get('data').shape)

# 1797, 64：有1797张图片，每张图片64个像素点，即64个特征向量
# pl.gray()  # 灰化
# pl.matshow(digits.images[0])
# pl.show()

X = digits.get('data')
y = digits.get('target')
# 将所有的X值转化到0-1之间
X -= X.min()  # normalize the values to bring them into the range 0-1
X /= X.max()
# print(X)

clf = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
)

# 将所有的y值转化为0,1的值
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

# print(labels_train)

# 开始训练
clf.fit(X_train, labels_train)

print(clf.n_layers_)  # 神经网络的层数，包括输入层，输出层
print(clf.n_outputs_)  # 输出的个数
print(clf.n_iter_)  # 迭代次数
print(clf.classes_)  # 每个输出的类标签

predictions = []
for i in range(X_test.shape[0]):
    o = clf.predict(X_test[i].reshape(1, -1))
    # 每一个样本进去，出来十中结果，选择其中一种比例高的，作为最终结果
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
