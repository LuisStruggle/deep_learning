#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matplotlib是类似matlab的一个可视化的画图包
分格显示
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 第一种方法 subplot2grid
# plt.figure()
# # 三行三列
# # 从第0行第0列开始，跨度3列，一行
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
# ax1.plot([1, 2], [1, 2])
# ax1.set_title('ax1_one')
# # 从第1行第0列开始，跨度2列，一行
# ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)

# # 从第1行第2列开始，跨度1列，两行
# ax4 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)

# # 从第2行第0列开始，跨度1列，一行
# ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)

# # 从第2行第1列开始，跨度1列，一行
# ax2 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)

# plt.show()

# 第二种方法 gridspec
# plt.figure()
# gs = gridspec.GridSpec(3, 3)
# ax1 = plt.subplot(gs[0, :])
# ax2 = plt.subplot(gs[1, :2])
# ax3 = plt.subplot(gs[1:, 2])
# ax4 = plt.subplot(gs[-1, 0])
# ax5 = plt.subplot(gs[-1, -2])

# plt.show()

# 第三种方法
# f就是画布对象figure，共享x轴，共享y轴
f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, sharex=True, sharey=True)

ax11.scatter([1, 2], [1, 2])

plt.show()
