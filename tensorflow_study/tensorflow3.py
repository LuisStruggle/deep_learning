#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

# tensorflow中定义变量
state = tf.Variable(0, name="zhang")

print(state.name)

# 定义一个常量
one = tf.constant(1)

new_value = tf.add(state, one)

update = tf.assign(state, new_value)

print(state.name)

init = tf.global_variables_initializer()  # 如果定义了变量，这句话必须有

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
