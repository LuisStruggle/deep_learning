#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用scikit-learn nerualnetwork简单实现CNN网络

"""
from sklearn.datasets import load_iris
# MLP（MultiLayer Perceptions）多层传感器（多层神经元）
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 获取数据集
digits = load_iris()

X = digits.get('data')
y = digits.get('target')

clf = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print(y_test)

# 开始训练
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)

score = clf.score(X_test, y_test)
print(score)
