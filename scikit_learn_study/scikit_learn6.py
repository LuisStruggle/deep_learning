#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持向量机（SVM）算法，线性不可区分
人脸识别案例
"""
import logging
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 打印出执行过程的全日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# 从数据集中得到人脸数据
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# print(lfw_people)

# introspect the images arrays to find the shapes (for plotting)
# 图像的形状参数
n_samples, h, w = lfw_people.images.shape

# 样本数量
print(n_samples)
# 暂时未知
# print(h)
# print(w)

X = lfw_people.get("data")
# print(X)
target_names = lfw_people.get("target_names")
# print(target_names)
y = lfw_people.get("target")
# print(y)

# 特征数量
n_features = X.shape[1]
print(n_features)
# 分类数量
n_classes = target_names.shape[0]
print(n_classes)

print("----------------------------------------------------------------------")

# Split into a training set and a test set using a stratified k fold
# 交叉验证方法（Cross-Validation）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print("----------------------------------------------------------------------")

# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components,
                                                          X_train.shape[0]))
t0 = time()
# n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目
# whiten ：判断是否进行白化，就是对降维后的数据的每个特征进行归一化
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

# eigenfaces（特征脸）
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("----------------------------------------------------------------------")
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
# 通过参数param_grid组合，选出一种一种结果最好的分类器
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
SVC()
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("----------------------------------------------------------------------")

# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

print("----------------------------------------------------------------------")

# Qualitative evaluation of the predictions using matplotlib


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.get_cmap('gray'))
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


prediction_titles = [
    title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])
]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
