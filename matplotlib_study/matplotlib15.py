#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
animation 动画
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

fig, ax = plt.subplots()
x = np.arange(0, 2 * np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i / 100))
    return line,


def init():
    line.set_ydata(np.sin(x))
    return line,


# frames是100帧，interval（间隔）
ani = animation.FuncAnimation(
    fig=fig, func=animate, frames=100, init_func=init, interval=20, blit=False)

plt.show()
