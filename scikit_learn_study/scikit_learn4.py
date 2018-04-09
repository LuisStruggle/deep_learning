#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持向量机（SVM）算法，线性可区分
"""

from sklearn import svm

x = [[2, 0], [1, 1], [2, 3]]  # 支持向量的定义
y = [0, 0, 1]  # 标记的定义

# 创建分类器
clf = svm.SVC(kernel='linear')

# 建模
clf.fit(x, y)

# 打印出支持向量
print(clf.support_vectors_)

# 打印出超平面两边，每一边各有几个支持向量
print(clf.n_support_)

# 预测一个点在超平面两边的分类

print(clf.predict([[1.5, 1.5]]))
print(clf.predict([[2, 2]]))
