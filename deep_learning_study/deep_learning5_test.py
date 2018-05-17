#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Theano实现的CNN
这个代码不再运行了
"""
import deep_learning5 as network
from deep_learning5 import Network
from deep_learning5 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network.load_data_shared()
mini_batch_size = 10

expanded_training_data, _, _ = network.load_data_shared(
    r"deep_learning_study/datasource/mnist_expanded.pkl")

net = Network([
    ConvPoolLayer(
        image_shape=(mini_batch_size, 1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2, 2),
        activation_fn=ReLU),
    ConvPoolLayer(
        image_shape=(mini_batch_size, 20, 12, 12),
        filter_shape=(40, 20, 5, 5),
        poolsize=(2, 2),
        activation_fn=ReLU),
    FullyConnectedLayer(
        n_in=40 * 4 * 4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(
        n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)
], mini_batch_size)
net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, validation_data,
        test_data)
"""
结果: 99.60% 显著提高
epochs: 减少到了40
隐藏层有 1000 个神经元
Ensemble of network: 训练多个神经网络, 投票决定结果, 有时会提高
"""
