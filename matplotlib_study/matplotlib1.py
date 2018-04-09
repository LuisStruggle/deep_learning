#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
"""
import matplotlib.pyplot as plt
import numpy as np

# 将-1到1之间平分成50分，返回50个点的坐标
x = np.linspace(-1, 1, 50)

print(x)

y = 2 * x + 1

plt.plot(x, y)

plt.show()

y = x**2

plt.plot(x, y)

plt.show()
