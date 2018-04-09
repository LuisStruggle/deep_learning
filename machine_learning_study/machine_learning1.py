#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统：简单的user-based协同过滤算法示例代码
"""
# 读取数据集

# 定义几种距离计算函数
# 更高效的方式为把得分向量化之后使用scipy中定义的distance方法
from math import sqrt


def euclidean_dis(rating1, rating2):
    """计算2个打分序列间的欧式距离. 输入的rating1和rating2都是打分dict
       格式为{'小时代4': 1.0, '疯狂动物城': 5.0}"""
    distance = 0
    commonRatings = False
    for key in rating1:
        if key in rating2:
            distance += (rating1[key] - rating2[key]) ^ 2
            commonRatings = True
    #两个打分序列之间有公共打分电影
    if commonRatings:
        return distance
    #无公共打分电影
    else:
        return -1


def manhattan_dis(rating1, rating2):
    """计算2个打分序列间的曼哈顿距离. 输入的rating1和rating2都是打分dict
       格式为{'小时代4': 1.0, '疯狂动物城': 5.0}"""
    distance = 0
    commonRatings = False
    for key in rating1:
        if key in rating2:
            distance += abs(rating1[key] - rating2[key])
            commonRatings = True
    #两个打分序列之间有公共打分电影
    if commonRatings:
        return distance
    #无公共打分电影
    else:
        return -1


def cos_dis(rating1, rating2):
    """计算2个打分序列间的cos距离. 输入的rating1和rating2都是打分dict
       格式为{'小时代4': 1.0, '疯狂动物城': 5.0}"""
    distance = 0
    dot_product_1 = 0
    dot_product_2 = 0
    commonRatings = False

    for score in rating1.values():
        dot_product_1 += score ^ 2
    for score in rating2.values():
        dot_product_2 += score ^ 2

    for key in rating1:
        if key in rating2:
            distance += rating1[key] * rating2[key]
            commonRatings = True
    #两个打分序列之间有公共打分电影
    if commonRatings:
        return 1 - distance / sqrt(dot_product_1 * dot_product_2)
    #无公共打分电影
    else:
        return -1


def pearson_dis(rating1, rating2):
    """计算2个打分序列间的pearson距离. 输入的rating1和rating2都是打分dict
       格式为{'小时代4': 1.0, '疯狂动物城': 5.0}"""
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x, 2)
            sum_y2 += pow(y, 2)
    # now compute denominator
    denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(
        sum_y2 - pow(sum_y, 2) / n)
    if denominator == 0:
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) / n) / denominator


#查找最近邻
def computeNearestNeighbor(username, users):
    """在给定username的情况下，计算其他用户和它的距离并排序"""
    distances = []
    for user in users:
        if user != username:
            #distance = manhattan_dis(users[user], users[username])
            distance = pearson_dis(users[user], users[username])
            distances.append((distance, user))
    # 根据距离排序，距离越近，排得越靠前
    distances.sort()
    return distances


#推荐
def recommend(username, users):
    """对指定的user推荐电影"""
    # 找到最近邻
    nearest = computeNearestNeighbor(username, users)[0][1]

    recommendations = []
    # 找到最近邻看过，但是我们没看过的电影，计算推荐
    neighborRatings = users[nearest]
    userRatings = users[username]
    for artist in neighborRatings:
        if not artist in userRatings:
            recommendations.append((artist, neighborRatings[artist]))
    results = sorted(
        recommendations, key=lambda artistTuple: artistTuple[1], reverse=True)
    for result in results:
        print result[0], result[1]