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
plt.plot(x, y)

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

# 添加标注
x0 = 1
y0 = 2 * x0 + 1
# 画个点，s是点的尺寸
plt.scatter(x0, y0, s=50, color='red')
# 给定两点画一条线lw是线的宽度
plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)

# 开始写标注*text，文本显示的位置，arrowprops是定义箭头，连线的样式
# arc是弧度，rad是角度
plt.annotate(
    r'2x+1=%s' % y0,
    xy=(x0, y0),
    xycoords='data',
    xytext=(+30, -30),
    textcoords='offset points',
    fontsize=16,
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

# 要显示中文，要输入fontproperties这个属性
plt.text(
    -1.7,
    3,
    u'this is text 可以有中文哦',
    fontproperties='SimHei',
    fontdict={
        'size': 12,
        'color': 'r'
    })

plt.show()
