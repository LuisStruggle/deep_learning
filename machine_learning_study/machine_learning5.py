#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取图像特征值
"""
# 载入CIFAR10数据集
from features import color_histogram_hsv, hog_feature
import numpy as np
from data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'julyedu/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

# 这里我们会抽取2种特征，第一是大家熟知的Histogram of Oriented Gradients (HOG)，第二种是HSV空间的颜色信息color_histogram_hsv
#
# 一般数来，HOG会捕获图片的纹理信息，而会忽略颜色信息，所以我们这里的color_histogram_hsv是对颜色的一个补充，我们希望2种信息咱们都能用上。

# 下面的hog_feature和color_histogram_hsv函数都只对一张图片做处理，抽取出相应的特征
# 而extract_features函数是一个批量的操作，得到一个特征矩阵，其中每一行都是对应位置的那张图片抽取出来的特征

num_color_bins = 10  # Number of bins in the color histogram
feature_fns = [
    hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)
]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# 依旧要做取均值的处理
mean_feat = np.mean(X_train_feats, axis=1)
mean_feat = np.expand_dims(mean_feat, axis=1)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# 然后咱们得做个标准化了，让所有的特征变化幅度都一致化
std_feat = np.std(X_train_feats, axis=1)
std_feat = np.expand_dims(std_feat, axis=1)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# 再加上我们的bias项
X_train_feats = np.vstack(
    [X_train_feats, np.ones((1, X_train_feats.shape[1]))])
X_val_feats = np.vstack([X_val_feats, np.ones((1, X_val_feats.shape[1]))])
X_test_feats = np.vstack([X_test_feats, np.ones((1, X_test_feats.shape[1]))])