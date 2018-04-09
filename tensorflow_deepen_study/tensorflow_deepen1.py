#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensorflow学习，helloworld
"""
import tensorflow as tf

# tensorflow 常量
hw = tf.constant("helloworld", dtype=tf.string)

# tensorflow 变量
variable1 = tf.Variable(initial_value='nihao')

init = tf.global_variables_initializer()

# 启动一个tensorflow的session（会话）
sess = tf.Session()

sess.run(init)  # 激活

# 运行Graph（计算图）
# tensorflow相当于一个客户端和服务器之间的联系，客户端是本地构建的代码模型，通过session创建的会话提交服务端运行，返回结果
print(sess.run(hw).decode())

print(sess.run(variable1).decode())

# 每一个tensor实例的运行都有一个图，如果没有显示定义图，则使用默认图
if hw.graph is tf.get_default_graph():
    print("true")

sess.close()
