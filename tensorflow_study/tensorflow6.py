#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
"""
生成可视化的结构图
"""

print(tf.VERSION)


def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    定义一个神经层
    """
    with tf.name_scope("layer"):
        with tf.name_scope("weight"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("tensorboard/logs", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)

writer.close()
sess.close()
