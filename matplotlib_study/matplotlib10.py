#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
3D数据
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 创建画布
fig = plt.figure()
# 加上3D的图
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
# 将X，Y mesh到3D图的底面
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)

# 高的值
Z = np.sin(R)

# 画3D的图rstride,cstride的跨度
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

# 加个等高线的图（从z轴压下去形成等高线）
ax.contourf(X, Y, Z, zdir='z', offiset=-2, cmap=plt.get_cmap('rainbow'))
# 设置z轴的坐标值
ax.set_zlim(-2, 2)

plt.show()
