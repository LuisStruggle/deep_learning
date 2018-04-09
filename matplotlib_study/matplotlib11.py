#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
Subplot 多合一显示
"""
import matplotlib.pyplot as plt

# plt.figure()
# # 创建小图（两行两列），1代表第一个位置
# plt.subplot(2, 2, 1)
# plt.plot([0, 1], [0, 1])
# # 创建小图（两行两列），2代表第二个位置
# plt.subplot(2, 2, 2)
# plt.plot([0, 1], [0, 2])
# # 创建小图（两行两列），3代表第三个位置
# plt.subplot(2, 2, 3)
# plt.plot([0, 1], [0, 3])
# # 创建小图（两行两列），4代表第四个位置
# plt.subplot(2, 2, 4)
# plt.plot([0, 1], [0, 4])

# 第二种排列

plt.figure()
# 创建小图（两行一列），1代表第一个位置，合并三列为一列
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])
# 创建小图（两行三列），4代表第二行的第一个位置
plt.subplot(2, 3, 4)
plt.plot([0, 1], [0, 2])
# 创建小图（两行三列），5代表第二行的第二个位置
plt.subplot(2, 3, 5)
plt.plot([0, 1], [0, 3])
# 创建小图（两行三列），6代表第二行的第三个位置
plt.subplot(2, 3, 6)
plt.plot([0, 1], [0, 4])

plt.show()
