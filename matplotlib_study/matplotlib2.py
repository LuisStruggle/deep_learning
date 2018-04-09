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
plt.plot(x, y2)
# 先的颜色，宽度，样式（虚线）
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

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

plt.show()
