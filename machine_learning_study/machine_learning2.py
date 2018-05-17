#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K最近邻/k-Nearest Neighbor (kNN)
"""
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from classifiers.k_nearest_neighbor import KNearestNeighbor
import sys

# 载入CIFAR-10数据集
cifar10_dir = r'machine_learning_study/dataset/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
"""
# 先了解一下训练集和测试集的维度等信息
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# 看看数据集中的一些样本：每个类别展示一些
classes = [
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck'
]
num_classes = len(classes)
samples_per_class = 7
plt.figure()
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    # 一个类别中挑出一些
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
# plt.show()
"""

# 咱们下采样一下，因为样本有点多哦，所以咱们只挑一部分，这样会更有效率一些
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# 咱们把图像数据展开成一个向量的形式
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)
"""
# 调用一下KNN分类器，然后训练(其实就是把样本都记下来)
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# 计算测试点和训练点之间的距离
dists = classifier.compute_distances_no_loops(X_test)

print(dists.shape)

# 可以可视化一下这个结果(虽然没啥意义)，矩阵中的每个点都代表一个测试样本和一个训练样本之间的距离
plt.figure()
plt.imshow(dists, interpolation='none')

# 其实预测阶段的过程非常简单，因为刚才和5000个训练样本之间的距离已经算好了，咱们直接取topN样本看类别就可以了
y_test_pred = classifier.predict_labels(dists, k=5)

# 看下准确率
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test,
                                               accuracy))
"""
""" 交叉验证
刚才的预测阶段，咱们的K是自己指定的5，实际上我们是可以通过交叉验证来确定最合适的K的 """

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

idxes = range(num_training)
idx_folds = np.array_split(idxes, num_folds)
for idx in idx_folds:
    #     mask = np.ones(num_training, dtype=bool)
    #     mask[idx] = False
    #     X_train_folds.append( (X_train[mask], X_train[~mask]) )
    #     y_train_folds.append( (y_train[mask], y_train[~mask]) )
    X_train_folds.append(X_train[idx])
    y_train_folds.append(y_train[idx])

k_to_accuracies = {}

classifier = KNearestNeighbor()
Verbose = False
for k in k_choices:
    if Verbose:
        print("processing k=%f" % k)
    else:
        sys.stdout.write('.')
    k_to_accuracies[k] = list()
    for num in range(num_folds):
        if Verbose:
            print("processing fold#%i/%i" % (num, num_folds))

        X_cv_train = np.vstack(
            [X_train_folds[x] for x in range(num_folds) if x != num])
        y_cv_train = np.hstack(
            [y_train_folds[x].T for x in range(num_folds) if x != num])

        X_cv_test = X_train_folds[num]
        y_cv_test = y_train_folds[num]

        # 训练KNN分类器
        classifier.train(X_cv_train, y_cv_train)

        # 计算和训练集之间图片的距离
        dists = classifier.compute_distances_no_loops(X_cv_test)

        y_cv_test_pred = classifier.predict_labels(dists, k=k)
        # 计算和预测
        num_correct = np.sum(y_cv_test_pred == y_cv_test)
        k_to_accuracies[k].append(float(num_correct) / y_cv_test.shape[0])

# 输出计算的准确率
plt.figure()
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

# 画一下结果
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array(
    [np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array(
    [np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# 根据上面交叉验证的结果，咱们确定最合适的k值为10，然后重新训练和测试一遍吧
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# 输出准确度
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test,
                                               accuracy))
