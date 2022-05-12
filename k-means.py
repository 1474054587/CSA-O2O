# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:28:47 2021

@author: 1111111
"""

import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt

# 循环更新簇中心
def cycle_updata_cluster(data, cluster_center, label, length, attribute, k):
    cluster_center_new = cluster_center.copy()
    while 1:
        for i in range(0, k):
            attribute_sum = np.random.randint(1, size=attribute)
            attribute_sum = attribute_sum.astype(float)
            cnt = 0
            for j in range(0, length):
                if label[j] == i:
                    cnt += 1
                    for m in range(0, attribute):
                        attribute_sum[m] = attribute_sum[m] + data[j][m]
            cluster = []
            for n in range(0, attribute):
                if cnt != 0:
                    attribute_sum[n] = float(attribute_sum[n]) / cnt
                    cluster.append(attribute_sum[n])
            cluster_center_new[i] = cluster
        if operator.eq(cluster_center_new, cluster_center):
            break
        cluster_center = cluster_center_new.copy()
        for i in range(0, length):
            dist = list()
            for j in range(0, k):
                if len(cluster_center_new[j]) > 0:
                    dist.append(get_distance(data[i], cluster_center_new[j], attribute, 1))
            label[i] = compare_distance(dist)
    return

# 比较距离大小，返回最小距离的标签
def compare_distance(dist):
    length = len(dist)
    min_index = 0
    for i in range(1, length):
        if dist[min_index] > dist[i]:
            min_index = i
    return min_index

# 计算两个点之间的距离
def get_distance(element, center, attribute, mode):
    dist = 0
    for i in range(0, attribute):
        dist += abs(element[i] - center[i]) * abs(element[i] - center[i])
    return math.sqrt(dist)

# 启动 kmeans
def kmeans(data, k):
    dataset = data.copy()
# 数据集的长度
    length = len(dataset)
# 数据集的属性维数
    attribute = len(dataset[0]) - 1
# 原数据集的标签
    origin = []
    for i in range(0, length):
        origin.append(dataset[i][attribute])
# 数据集
    data_set = np.delete(dataset, attribute, axis=1).tolist()
# 聚类后的标签
    label = list()
# 用kmeans++确定初始簇中心
    cluster_center = []
    rand_x = np.random.randint(0, length)
    cluster_center.append(data_set[rand_x])
    for i in range(0, k - 1):
        dist = []
        for j in range(0, length):
            dist.append(get_distance(data_set[j], cluster_center[i], attribute, 1))
        prob = dist / np.sum(dist) # 该点当选聚类中心的概率
        rand_x = np.random.choice(length, p=prob) # 用此概率来进行随机
        cluster_center.append(data_set[rand_x])
# 初始化数据集和簇中心
    for i in range(0, length):
        dist = list()
        for j in range(0, k):
            dist.append(get_distance(data_set[i], cluster_center[j], attribute, 1))
        label.append(compare_distance(dist))
# 循环更新簇中心与每个点的标签
    cycle_updata_cluster(data_set, cluster_center, label, length, attribute, k)
# 画出聚类后的分布图
    color = ['red', 'green', 'blue', 'black']
    for i in range(0, k):
        plot_x = []
        plot_y = []
        for j in range(0, length):
            if label[j] == i:
                x = 1.0
                for m in range(0, attribute):
                    x += data[j][m]
                plot_x.append(data[j][1])
                plot_y.append(float(x / attribute))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('散点图')
        str_label = '类别' + str(i + 1)
        plt.scatter(plot_x, plot_y, c=color[i], alpha=1, label=str_label)
    plt.grid(True)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend(loc='best')
    plt.show()
    


# 程序开始

#1. iris 数据集的预处理
def iris_deal(v):
    if v == "Iris-setosa":
        v = 1
    elif v == "Iris-versicolor":
        v = 2
    elif v == "Iris-virginica":
        v = 3
    return v
#1. iris 数据集
iris = pd.read_csv("C:\\Users\\1111111\\Desktop\\CSA\\test\\iris.data", header=None)
iris.iloc[:, 4] = iris.iloc[:, 4].apply(iris_deal)
iris = np.array(iris)
iris = iris.tolist()
kmeans(iris, 2)

#2. wine 数据集
wine = pd.read_csv("C:\\Users\\1111111\\Desktop\\CSA\\test\\wine.data", header=None)
wine = np.array(wine)
wine = wine.tolist()
kmeans(wine, 3)

#3. car 数据集 前四列的预处理
def car_deal_1(v):
    if v == "vhigh":
        v = 1
    elif v == "high":
        v = 2
    elif v == "med":
        v = 3
    elif v == "low":
        v = 4
    elif v == "5more":
        v = 5
    elif v == "more":
        v = 6
    else : v=0
    return v
#3. car 数据集 后两列（第7列无用，删去）的预处理
def car_deal_2(v):
    if v == "small":
        v = 1
    elif v == "low":
        v = 1
    elif v == "med":
        v = 2
    elif v == "big":
        v = 3
    elif v == "high":
        v = 3
    elif v == "unacc":
        v = 1
    elif v == "acc":
        v = 2
    else : v= 0
    return v
#3. car 数据集
car = pd.read_csv("C:\\Users\\1111111\\Desktop\\CSA\\test\\car.data", header=None)
car.iloc[:, 0] = car.iloc[:, 0].apply(car_deal_1)
car.iloc[:, 1] = car.iloc[:, 1].apply(car_deal_1)
car.iloc[:, 2] = car.iloc[:, 2].apply(car_deal_1)
car.iloc[:, 3] = car.iloc[:, 3].apply(car_deal_1)
car.iloc[:, 4] = car.iloc[:, 4].apply(car_deal_2)
car.iloc[:, 5] = car.iloc[:, 5].apply(car_deal_2)
car.drop([6], axis=1, inplace=True)
car = np.array(car)
car = car.tolist()
kmeans(car, 3)

#4. abalone 数据集的预处理
def abalone_deal(v):
    if v == "F":
        v = 1
    elif v == "M":
        v = 2
    elif v == "I":
        v = 3
    return v
#4. abalone 数据集
abalone = pd.read_csv("C:\\Users\\1111111\\Desktop\\CSA\\test\\abalone.data", header=None)
abalone.iloc[:, 0] = abalone.iloc[:, 0].apply(abalone_deal)
abalone = np.array(abalone)
abalone = abalone.tolist()
kmeans(abalone, 3)

#5. forestfires 数据集的预处理
def forestfires_deal(v):
    if v == "jan":
        v = 1
    elif v == "feb":
        v = 2
    elif v == "mar":
        v = 3
    elif v == "apr":
        v = 4
    elif v == "may":
        v = 5
    elif v == "jun":
        v = 6
    elif v == "jul":
        v = 7
    elif v == "aug":
        v = 8
    elif v == "sep":
        v = 9
    elif v == "oct":
        v = 10
    elif v == "nov":
        v = 11
    elif v=="dec":
        v = 12
    return v
#5. forestfires 数据集
forestfires = pd.read_csv("C:\\Users\\1111111\\Desktop\\CSA\\test\\forestfires.csv", header=None)
forestfires.drop([0], inplace=True)
forestfires.iloc[:, 2] = forestfires.iloc[:, 2].apply(forestfires_deal)
forestfires.drop([3], axis=1, inplace=True)
forestfires[0] = forestfires[0].map(float)
forestfires[1] = forestfires[1].map(float)
forestfires[4] = forestfires[4].map(float)
forestfires[5] = forestfires[5].map(float)
forestfires[6] = forestfires[6].map(float)
forestfires[7] = forestfires[7].map(float)
forestfires[8] = forestfires[8].map(float)
forestfires[9] = forestfires[9].map(float)
forestfires[10] = forestfires[10].map(float)
forestfires[11] = forestfires[11].map(float)
forestfires[12] = forestfires[12].map(float)
forestfires = np.array(forestfires)
forestfires = forestfires.tolist()
kmeans(forestfires, 4)
