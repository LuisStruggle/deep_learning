#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策树练习
"""
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree

allElectronicsData = open('scikit_learn_study/datasource/AllElectronics.csv',
                          'r')
reader = csv.reader(allElectronicsData)
headers = reader.__next__()

print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])  # 将每一行的标记放入列表中
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(labelList)
print(featureList)

print("----------------------------------------------------------------------")

# Vectorizer　featureList（矢量化字典列表，目的是为了得到训练集）
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print(dummyX)
print(vec.get_feature_names())
print("----------------------------------------------------------------------")
# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

print(dummyY)
print("----------------------------------------------------------------------")

# 创建分类器
clf = tree.DecisionTreeClassifier(
    criterion="entropy")  # criterion="entropy"，使用信息熵来确定不确定性最大的变量
# 建模，构建信息树
clf = clf.fit(dummyX, dummyY)

print(clf)
print("----------------------------------------------------------------------")
# Visualize model（将这个决策树输出到文件中
with open("scikit_learn_study/datasource/allElectronicInformationGainOri.dot",
          'w') as f:
    f = tree.export_graphviz(
        clf, feature_names=vec.get_feature_names(), out_file=f)

# 读取训练集中的第一行数据，作为测试数据
oneRowX = dummyX[0]
print(oneRowX)

# 修改一下第一行的原始数据
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0

print(newRowX)

predictedY = clf.predict(newRowX.reshape(1, -1))
print(predictedY)  # 预测的结果为1
