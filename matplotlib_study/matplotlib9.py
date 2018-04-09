#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
Image图片
"""
import matplotlib.pyplot as plt
import numpy as np

a = np.array([
    0.313660827978, 0.365348418405, 0.423733120134, 0.365348418405,
    0.439599930621, 0.525083754405, 0.423733120134, 0.525083754405,
    0.651536351379
]).reshape((3, 3))

# interpolation 填充样式
plt.imshow(
    a, interpolation='nearest', cmap=plt.get_cmap('hot'), origin='lower')

# 颜色标注
plt.colorbar()

plt.xticks(())
plt.yticks(())

plt.show()
