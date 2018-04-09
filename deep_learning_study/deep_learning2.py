#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从sklearn导入mnist数据集

mnist_svm

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier.
"""
from sklearn import svm
import pickle


def svm_baseline():
    with open('deep_learning_study/datasource/mnist.pkl', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(
            f, encoding='iso-8859-1')
        # train
        clf = svm.SVC()
        clf.fit(training_data[0][:10000], training_data[1][:10000])
        # test
        predictions = clf.predict(test_data[0][:2000])
        num_correct = sum(
            int(a == y) for a, y in zip(predictions, test_data[1][:2000]))

        print("Baseline classifier using an SVM.")
        print("%s of %s values correct." % (num_correct,
                                            len(test_data[1][:2000])))


if __name__ == "__main__":
    svm_baseline()
