#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
等高线图
"""
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)


n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)

# 将x,y定义成网格
X, Y = np.meshgrid(x, y)
print(X)
print(Y)

# 填充等高线图的画布（cmap=color map）
plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.get_cmap('hot'))
# 画出等高线
C = plt.contour(X, Y, f(X, Y), 8, color='black', linewidth=0.5)
# 添加lable（clabel，就是等高线的label）
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()
