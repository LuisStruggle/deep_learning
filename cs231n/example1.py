#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest Neighbor classifier
"""
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        X is N x D，每一行一个样本，y是一个的尺寸为N的列表
        """
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        X is N x D where each row is an example we wish to predict label for
        """
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # 没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(
                distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[
                min_index]  # predict the label of the nearest example

        return Ypred


mnist = fetch_mldata(
    'MNIST Original', data_home='deep_learning_study/datasource')
"""
train_data：所要划分的样本特征集；train_target：所要划分的样本结果；
test_size：样本占比，如果是整数的话就是样本的数量；random_state：是随机数的种子。
（随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。）
"""
X_train, X_test, y_train, y_test = train_test_split(
    mnist.get('data'), mnist.get('target'), test_size=0.25, random_state=0)

nn = NearestNeighbor()
nn.train(X_train[0:5000, :], y_train[0:5000])

rs = nn.predict(X_test[0:300, :])

# 正确率
zql = np.sum([1 for i, j in zip(rs, y_test[0:300]) if i == j])

print(zql/300)
