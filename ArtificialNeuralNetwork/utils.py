# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import random


__author__ = 'wuyueqiu'


def gen_data():
    # 生成数据
    x1 = [random.random() * 4 - 2 for _ in range(50)]
    x2 = [np.sqrt(1 - 0.25 * i ** 2) - 0.5 * random.random() for i in x1]
    x2 += [-np.sqrt(1 - 0.25 * i ** 2) + 0.5 * random.random() for i in x1]
    x1 = x1 * 2
    X1 = np.array([[x1[i], x2[i]] for i in range(len(x1))])
    y1 = np.array([0 for _ in range(100)])

    x1 = [random.random() * 6 - 3 for _ in range(50)]
    x2 = [np.sqrt(4 - 4 / 9 * i ** 2) - 0.5 * random.random() for i in x1]
    x2 += [-np.sqrt(4 - 4 / 9 * i ** 2) + 0.5 * random.random() for i in x1]
    x1 = x1 * 2
    X2 = np.array([[x1[i], x2[i]] for i in range(len(x1))])
    y2 = np.array([1 for _ in range(100)])

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    y = y.reshape(y.size, 1)
    return X, y


def show_data(X, y):
    X1 = X[y[:, 0] >= 0.5]
    X2 = X[y[:, 0] < 0.5]
    plt.plot(X1[:, 0], X1[:, 1], 'bo')
    plt.plot(X2[:, 0], X2[:, 1], 'rx')
    plt.show()


if __name__ == '__main__':
    X, y = gen_data()
    print(X)
    print(y)
    show_data(X, y)
