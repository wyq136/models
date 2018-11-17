# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import random


__author__ = 'wuyueqiu'


def gen_data():
    # 生成数据
    X1 = [[1 + random.random(), 1 + random.random()] for i in range(50)]
    X2 = [[2 + random.random(), 2 + random.random()] for i in range(50)]
    y1 = [0 for i in range(50)]
    y2 = [1 for i in range(50)]
    X = np.array(X1 + X2)
    y = np.array(y1 + y2)
    return X, y


if __name__ == '__main__':
    X, y = gen_data()
    print(X)
    print(y)
    plt.plot(X[:50, 0], X[:50, 1], 'bo')
    plt.plot(X[50:, 0], X[50:, 1], 'rx')
    plt.show()
