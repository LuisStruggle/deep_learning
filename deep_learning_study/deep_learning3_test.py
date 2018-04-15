#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import deep_learning3 as network
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

# 加载训练数据和测试数据
mnist = fetch_mldata(
    'MNIST Original', data_home='deep_learning_study/datasource')

data = mnist.get('data')
# 将data格式化成0,1数组
data = data / 255
target = mnist.get('target')

# 将数据分为train_data和test_data
train_x, test_x, train_y, test_y = train_test_split(data, target)

# 将输入数据格式化为神经网络运行的格式
train_x = [np.reshape(x, (784, 1)) for x in train_x]
test_x = [np.reshape(x, (784, 1)) for x in test_x]
train_y = LabelBinarizer().fit_transform(train_y).reshape(-1, 10, 1)

train_data = list(zip(train_x, train_y))
test_data = list(zip(test_x, test_y))

net = network.Network([784, 30, 10])
net.SGD(train_data, 30, 10, 3.0, test_data=test_data)
