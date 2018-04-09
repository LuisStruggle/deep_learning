#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持向量机（SVM）算法，线性可区分
"""
import numpy as np
import pylab as pl
from sklearn import svm

# seed值为0，第一次随机产生的数，以后再重新运行程序不在变化，值为1，则每次重新运行都产生新的数值
np.random.seed(0)

# 随机产生支持向量，20行两列
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20  # 标记的定义

# 创建分类器
clf = svm.SVC(kernel='linear')

# 建模、
clf.fit(x, y)

# get the separating hyperplane（w0x+w1y+w3=0）转化为-（w0/w1）x-(w3/w1)=y
w = clf.coef_[0]
a = -w[0] / w[1]  # 斜率-(w0/w1)=a
xx = np.linspace(-5, 5)
# clf.intercept_[0])是w3偏差（bias），即截距（intercept）
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors（求上下两条边界线）
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print("w: ", w)
print("a: ", a)
print("xx: ", xx)
print("yy: ", yy)
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_: ", clf.coef_)

# 通过pl画图
# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')  # xx值，yy值，最后一个参数是线条的表示形式
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

# 画出生成的40个支持向量的点，s表示点点的大小，c就是color嘛，marker就是点点的形状o,x,*><^,
# alpha是点点的亮度
pl.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=30,
    c='red',
    marker='o',
    alpha=0.5)
pl.scatter(x[:, 0], x[:, 1], s=30, c='blue', marker='x', alpha=0.5)

# axis tight是使坐标系的最大值和最小值和你的数据范围一致！
pl.axis('tight')
pl.show()
