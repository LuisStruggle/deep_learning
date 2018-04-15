#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非监督学习算法（无类别标记(class label)）：聚类(clustering)：hierarchical clustering 层次聚类
"""
from PIL import Image, ImageDraw
from scikit_learn14 import hcluster
from scikit_learn14 import getheight
from scikit_learn14 import getdepth
import numpy as np
import os


def drawdendrogram(clust, imlist, jpeg='clusters.jpg'):
    h = getheight(clust) * 1000
    w = 1200
    depth = getdepth(clust)

    scaling = float(w - 150) / depth

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.line((0, h / 2, 50, h / 2), fill=(255, 0, 0))

    drawnode(draw, clust, 50, int(h / 2), scaling, imlist, img)
    img.save(r'scikit_learn_study\datasource\image\\' + jpeg)


def drawnode(draw, clust, x, y, scaling, imlist, img):
    if clust.id < 0:
        h1 = getheight(clust.left) * 200
        h2 = getheight(clust.right) * 200
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2

        ll = clust.distance * scaling

        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))
        draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))
        draw.line(
            (x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))

        drawnode(draw, clust.left, x + ll, top + h1 / 2, scaling, imlist, img)
        drawnode(draw, clust.right, x + ll, bottom - h2 / 2, scaling, imlist,
                 img)
    else:
        nodeim = Image.open(imlist[clust.id])
        nodeim.thumbnail((50, 50))
        ns = nodeim.size
        print(x, y - ns[1] // 2)
        print(x + ns[0])
        print(
            img.paste(nodeim, (int(x), int(y - ns[1] // 2), int(x + ns[0]),
                               int(y + ns[1] - ns[1] // 2))))
        # img.paste()


imlist = []
folderPath = r'scikit_learn_study\datasource\image'
for filename in os.listdir(folderPath):
    if os.path.splitext(filename)[1] == '.jpg':
        imlist.append(os.path.join(folderPath, filename))

n = len(imlist)
print(n)

features = np.zeros((n, 3))
# a是个矩阵或者数组，a.flatten()就是把a降到一维，默认是按横的方向降
for i in range(n):
    im = np.array(Image.open(imlist[i]))
    R = np.mean(im[:, :, 0].flatten())
    G = np.mean(im[:, :, 1].flatten())
    B = np.mean(im[:, :, 2].flatten())
    features[i] = np.array([R, G, B])

tree = hcluster(features)
drawdendrogram(tree, imlist, jpeg='sunSet.jpg')
