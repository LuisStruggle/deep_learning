#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
# 下载数据
from tensorflow.examples.tutorials.mnist import input_data

# one_hot 是独热码的编码（encoding）形式
# 即把下载下来的标记0,1,2,3,4,5,6,7,8,9 编码为0:1000000000,1:0100000000
mnist = input_data.read_data_sets("tensorflow_study/MNIST_data", one_hot=True)

# 行未知，列为28*28
input_x = tf.placeholder(tf.float32, (None, 28 * 28)) / 255
output_y = tf.placeholder(tf.float32, (None, 10))

# 改变input_x数据形状，变成长宽高28*28*1的一个格式,-1表示多少行，不清楚
input_x_images = tf.reshape(input_x, (-1, 28, 28, 1))

# 提取测试集数据
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

# 构建我们的卷积神经网络
# 第一层卷积 strides：步长，padding=same，表示输出的大小不变，输出还是28*28，在外围补零两圈
conv1 = tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=(5, 5),
    padding="same",
    strides=1,
    activation=tf.nn.relu)

# 第一层pool
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)

# 第二层卷积 strides：步长，padding=same，表示输出的大小不变，14*14*64
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    strides=1,
    activation=tf.nn.relu)

# 第二层pool（7*7*64）
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2)

# 平坦化(-1,表示根据之后的确定的参数去推断-1这个位置上的维度大小)
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# 1024个神经元的全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout改进算法，避免overfitting，丢弃50%的数据
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# 10个全连接层的输出，这里不用激活函数来做非线性化了
logits = tf.layers.dense(inputs=dropout, units=10)

# 计算误差（计算 Cross entropy（交叉熵），再用 Softmax 计算百分比概率）
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# Adam 优化器来最小化误差，学习率 0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 精度。计算 预测值 和 实际标签 的匹配程度
# 返回(accuracy, update_op),
# argmax返回的是最大数的索引.argmax有一个参数axis,默认是0,表示第几维的最大值
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(input=output_y, axis=1),
    predictions=tf.argmax(input=logits, axis=1))[1]

# 创建会话
sess = tf.Session()
# 初始化变量：全局和局部
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

sess.run(init)

for i in range(2000):
    # 从 Train（训练）数据集里取“下一个” 50 个样本
    batch = mnist.train.next_batch(50)
    train_loss, train_op_ = sess.run(
        fetches=(loss, train_op),
        feed_dict={
            input_x: batch[0],
            output_y: batch[1]
        })

    if i % 100 == 0:
        test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
        print("Step=%d, Train loss=%.4f, [Test accuracy=%.2f]" %
              (i, train_loss, test_accuracy))

# 测试：打印 20 个预测值 和 真实值 的对
test_output = sess.run(logits, feed_dict={input_x: test_x[:20]})
inferenced_y = tf.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers')  # 推测的数字
print(tf.argmax(test_y[:20], 1), 'Real numbers')  # 真实的数字
