#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
图中图
"""
import matplotlib.pyplot as plt

fig = plt.figure()
x = [x for x in range(1, 8)]
y = [1, 3, 4, 2, 5, 8, 6]

# 添加第一个大图left, bottom, width, height，是图的一个定位
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

# 添加第二个图
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x, y, 'b')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title inside 1')

# 添加第三个图
plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(x, y, 'y')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title inside 2')

plt.show()
