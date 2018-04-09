#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
柱状图
"""
import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1)
plt.bar(X, -Y2)

plt.xlim(-5, n)
plt.xticks(())

plt.ylim(-1.25, 1.25)
plt.yticks(())

# 相对于定义的x，y点，ha水平位置，va垂直位置
for x, y in zip(X, Y1):
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, -Y2):
    plt.text(x + 0.4, y - 0.05, '-%.2f' % y, ha='center', va='top')

plt.show()
