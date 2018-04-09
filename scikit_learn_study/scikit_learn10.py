#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监督学习算法：多元线性回归Regression
"""
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelBinarizer

ds = np.array([[100, 4, 1, 9.3], [50, 3, 0, 4.8], [100, 4, 1, 8.9],
               [100, 2, 2, 6.5], [50, 2, 2, 4.4], [80, 2, 1, 6.2],
               [75, 3, 1, 7.4], [65, 4, 0, 6.0], [90, 3, 0,
                                                  7.6], [90, 2, 2, 6.1]])

X = np.append(ds[:, :-2], LabelBinarizer().fit_transform(ds[:, -2]), axis=1)
Y = ds[:, -1]

print(X)
print(Y)

regr = linear_model.LinearRegression()

regr.fit(X, Y)

print("coefficients:", regr.coef_)  # x前的各个系数
print("intercept:", regr.intercept_)  # 截距

xPred = [102, 6, 0, 1, 0]
yPred = regr.predict(np.array(xPred).reshape(1, -1))
print("predicted y:", yPred)
