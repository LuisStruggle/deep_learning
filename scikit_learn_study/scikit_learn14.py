#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非监督学习算法（无类别标记(class label)）：聚类(clustering)：hierarchical clustering 层次聚类
"""
import numpy as np
import math


class cluster_node:
    def __init__(self,
                 vec,
                 left=None,
                 right=None,
                 distance=0.0,
                 id=None,
                 count=1):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance
        self.count = count  # only used for weighted average


# 第一种距离计算方式
def L2dist(v1, v2):
    # 平方和，开根号
    return math.sqrt(sum((v1 - v2)**2))


# 第二种距离计算方式
def L1dist(v1, v2):
    return sum(abs(v1 - v2))


def hcluster(features, distance=L2dist):
    # 将特征矩阵的行聚类
    # cluster the rows of the "features" matrix
    distances = {}
    currentclustid = -1

    # clusters are initially just the individual rows
    # 初始化树，每一个节点是一个聚类
    clust = [
        cluster_node(np.array(features[i]), id=i)
        for i in range(len(features))
    ]

    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(
                        clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        # calculate the average of the two clusters
        mergevec = [
            (clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0
            for i in range(len(clust[0].vec))
        ]

        # create the new cluster
        newcluster = cluster_node(
            np.array(mergevec),
            left=clust[lowestpair[0]],
            right=clust[lowestpair[1]],
            distance=closest,
            id=currentclustid)

        # cluster ids that weren't in the original set are negative
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]


# 提取聚类结果
def extract_clusters(clust, dist):
    # extract list of sub-tree clusters from hcluster tree with distance<dist
    # clusters = {}
    if clust.distance < dist:
        # we have found a cluster subtree
        return [clust]
    else:
        # check the right and left branches
        cl = []
        cr = []
        if clust.left is not None:
            cl = extract_clusters(clust.left, dist=dist)
        if clust.right is not None:
            cr = extract_clusters(clust.right, dist=dist)
        return cl + cr


def get_cluster_elements(clust):
    # return ids for elements in a cluster sub-tree
    if clust.id >= 0:
        # positive id means that this is a leaf
        return [clust.id]
    else:
        # check the right and left branches
        cl = []
        cr = []
        if clust.left is not None:
            cl = get_cluster_elements(clust.left)
        if clust.right is not None:
            cr = get_cluster_elements(clust.right)
        return cl + cr


def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n):
        print(' ')
    if clust.id < 0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels is None:
            print(clust.id)
        else:
            print(labels[clust.id])

    # now print the right and left branches
    if clust.left is not None:
        printclust(clust.left, labels=labels, n=n + 1)
    if clust.right is not None:
        printclust(clust.right, labels=labels, n=n + 1)


def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left is None and clust.right is None:
        return 1

    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left is None and clust.right is None:
        return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance


if __name__ == "__main__":
    clust = hcluster([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(getheight(clust))
    # print(getdepth(clust))
    # print(printclust(clust))
    # print(get_cluster_elements(clust))
    print(extract_clusters(clust, 7))
