#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 简单的张量分解进行打分和推荐
# 要用到numpy模块
import numpy


# 手写矩阵分解
# 现在有很多很方便对高维矩阵做分解的package，比如libmf, svdfeature等
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (
                            2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (
                            2 * eij * P[i][k] - beta * Q[k][j])

        # eR = numpy.dot(P, Q)
        # 优化误差，当误差小于0.001时，结束优化
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (
                            pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


# 读取user数据并用张量分解进行打分

R = [
    [5, 3, 0, 1],
    [4, 0, 3, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
]

R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 2

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)

print('nP=', nP)

print('nQ=', nQ)

print('nR=', nR)

print('R=', R)
