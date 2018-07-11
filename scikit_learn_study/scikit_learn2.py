#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最近邻规则分类KNN算法练习
"""
from sklearn import neighbors
from sklearn import datasets

# 获取iris数据集
iris = datasets.load_iris()

# print(iris)

# 创建分类器
ng = neighbors.KNeighborsClassifier()

# 建模
ng.fit(iris.get("data"), iris.get("target"))

result = ng.predict([[0.1, 0.2, 0.3, 0.4]])

print(result)
