#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加上了一些flags
加上了save
"""
import deep_learning4 as network
import mnist_data

training_data, validation_data, test_data = mnist_data.mnist_data()
print(len(training_data))
print(len(validation_data))
print(len(test_data))
###############################################################################
# 以上是数据处理，下面是网络训练
net = network.Network([784, 100, 10], cost=network.CrossEntropyCost)

net.SGD(
    training_data,
    30,
    10,
    0.1,
    5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_accuracy=True,
    monitor_training_cost=True)
