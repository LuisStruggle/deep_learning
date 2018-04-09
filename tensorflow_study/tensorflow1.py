#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过数据预测一个函数的常数项和未知数x的系数（y=ax+b），即预测a，b的值
"""
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float)

y_data = x_data * 0.1 + 0.3

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 对应0.1的一个预测值
biases = tf.Variable(tf.zeros([1]))  # 对应0.3的一个预测值

y = Weights * x_data + biases  # 当前预测的y值

loss = tf.reduce_mean(tf.square(y - y_data))  # 当前预测的y值和实际值y_data之间的方差
optimizer = tf.train.GradientDescentOptimizer(
    0.5)  # 创建一个优化器，0.5是一个学习效率，一般是一个小于1的数
train = optimizer.minimize(loss)  # 训练优化，减少误差

init = tf.global_variables_initializer()
# create tensorflow structure end

sess = tf.Session()
sess.run(init)  # 激活

# 开始用数据训练
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
