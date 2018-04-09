#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)

y = 2 * x + 1

# figure 创建一个新画布，到下一个figure出现之前，当前的figure就结束了
plt.figure()
plt.plot(x, y, linewidth=10, zorder=1)

# 对坐标轴进行操作
ax = plt.gca()
# 去掉坐标轴上面的线和右边的线
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 设置坐标轴原点的位置
# 用下面的坐标轴代替我的x轴，图默认没有x，y轴，下面两步设置指定x，y轴
ax.xaxis.set_ticks_position('bottom')
# 用左边的坐标轴代替我的y轴
ax.yaxis.set_ticks_position('left')
# 定义坐标轴的位置
# 设置横坐标的位置，是y轴线上的0位置
ax.spines['bottom'].set_position(('data', 0))  # outward,axes
# 设置纵坐标的位置，是x轴线上的0位置
ax.spines['left'].set_position(('data', 0))

# 当线条挡住了我们的坐标轴上的数值时，用下面的方法，zorder设置线的显示层级，alpha是透明度
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(
        dict(facecolor='white', edgecolor='None', alpha=0.7, zorder=2))

plt.show()
