#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
衡量两个值的线性相关强度的量（衡量线性回归，如果效果不是特别的好，则可以选择非线性回归算法）：回归中的相关度和R平方值
"""
import numpy as np
import math


# Correlation（相关性计算）
def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2

    SST = math.sqrt(varX * varY)
    return SSR / SST


testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

print('r:', computeCorrelation(testX, testY))

print('简单线性回归的决定系数r^2:', computeCorrelation(testX, testY)**2)


# 多元线性回归的决定系数r^2
# Polynomial Regression
def polyfit(x, y, degree=1):
    results = {}
    # 这个函数会自动计算x，y之间的线性关系的截距，斜率
    coeffs = np.polyfit(x, y, degree)  # degree指的是x的最高次幂

    # Polynomial Coefficients（多项式的系数）
    # 将coeffs系数对象转化为一个list对象
    results['polynomial'] = coeffs.tolist()

    # r-squared
    # 通过多项式系数，构建一个一维的模型
    p = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p(x)  # 通过模型给定自变量，预测它的y是多少
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['determination'] = ssreg / sstot

    return results


print('# 多元线性回归的决定系数r^2:', polyfit(testX, testY))
