#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展mnist数据集，方法：通过移动每个训练图像向上、向下、向左和

对，一个像素

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.
Note that this program is memory intensive, and may not run on small
systems.
"""
from __future__ import print_function

import pickle
import os.path
import random

import numpy as np

print("Expanding the MNIST training set")

if os.path.exists("deep_learning_study/datasource/mnist_expanded.pkl"):
    print("The expanded training set already exists.  Exiting.")
else:
    with open('deep_learning_study/datasource/mnist.pkl', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(
            f, encoding='iso-8859-1')

    expanded_training_pairs = []
    j = 0  # counter
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0:
            print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, index_position, index in [(1, 0, "first",
                                                0), (-1, 0, "first",
                                                     27), (1, 1, "last", 0),
                                               (-1, 1, "last", 27)]:
            # np.roll(a, shift, axis=None)
            # 意思是将a，沿着axis的方向，滚动shift长度
            new_img = np.roll(image, d, axis)
            if index_position == "first":
                new_img[index, :] = np.zeros(28)
            else:
                new_img[:, index] = np.zeros(28)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))

    # 打乱数据集
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("Saving expanded data. This may take a few minutes.")
    with open("deep_learning_study/datasource/mnist_expanded.pkl", "wb") as f:
        pickle.dump((expanded_training_data, validation_data, test_data), f)
