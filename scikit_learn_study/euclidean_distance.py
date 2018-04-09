#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欧几里得距离，即求两个点之间的直线距离
"""
import math


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


print(euclideanDistance(3, 4, 0, 0))
