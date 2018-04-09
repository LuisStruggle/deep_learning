#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")

c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")

e = tf.add(c, d, name="add_e")

sess = tf.Session()
# sess.run(e)
output = sess.run(e)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tensorboard/logs', sess.graph)

writer.close()
sess.close()
