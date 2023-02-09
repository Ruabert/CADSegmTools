"""
Copy from https://blog.csdn.net/zkt286468541/article/details/105550948

@author: alphaTao
"""
import os.path
import numpy as np


# 计算欧式距离
def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 判断两点是否在范围内
def isNeighbor(x, y, eps):
    return distEuclid(x, y) <= eps


# 获取某一点邻域内的点
def getSeedPos(pos, data, eps):
    seed = []
    for p in range(len(data)):
        if isNeighbor(data[p], data[pos], eps):
            seed.append(p)
    return seed


# 获取核心点列表
def getCorePointsPos(data, eps, minpts):
    cpoints = []
    for pos in range(len(data)):
        if len(getSeedPos(pos, data, eps)) >= minpts:
            cpoints.append(pos)
    return cpoints


# 分类
def getClusters(data, eps, minpts):
    corePos = getCorePointsPos(data, eps, minpts)
    unvisited = list(range(len(data)))
    cluster = {}
    num = 0

    for pos in corePos:
        if pos not in unvisited:
            continue
        clusterpoint = []
        clusterpoint.append(pos)
        seedlist = getSeedPos(pos, data, eps)
        unvisited.remove(pos)
        while seedlist:
            p = seedlist.pop(0)
            if p not in unvisited:
                continue
            unvisited.remove(p)
            clusterpoint.append(p)
            if p in corePos:
                seedlist.extend(getSeedPos(p, data, eps))
        cluster[num] = clusterpoint
        num += 1
    cluster["noisy"] = unvisited
    return cluster













