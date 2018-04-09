#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)

y1 = 2 * x + 1

y2 = x**2

# figure 创建一个新画布，到下一个figure出现之前，当前的figure就结束了
plt.figure(num=4, figsize=(8, 5))
plt.plot(x, y1)

plt.figure()
l1, = plt.plot(x, y2, label='up')
# 先的颜色，宽度，样式（虚线）
l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')

# 设置坐标轴标尺的显示范围
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# 设置坐标轴的标签
plt.xlabel('I am x')
plt.ylabel('I am y')

new_ticks = np.linspace(-1, 2, 5)
# 更换坐标的显示值（标尺）
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3],
           ['really bad', 'bad', 'normal', 'good', 'really good'])

# 打印图例，前提是plot里要有label这个属性
# loc就是位置，要是没有labels，就用label后面的标签
plt.legend(handles=[l1, l2], labels=['aa', 'bb'], loc='best')

plt.show()
