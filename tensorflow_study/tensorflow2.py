#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

# 两个矩阵相乘
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)  # np.dot(m1,m2)也可以实现
""" sess = tf.Session()

result = sess.run(product)

print(result)

sess.close() """

# 自动close
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
