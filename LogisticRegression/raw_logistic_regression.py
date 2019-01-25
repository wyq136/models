# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import utils


__author__ = 'wuyueqiu'

X, y = utils.gen_data()

# 定义参数 w 和 b
theta = np.random.rand(2)
bias = 0

learning_rate = 1
for epoch in range(1000):
    # 定义模型，前向计算
    z = X.dot(theta) + bias
    pred_y = 1 / (1 + np.exp(-z))

    # loss
    loss = - (y * np.log(pred_y) + (1 - y) * np.log(1 - pred_y)).mean()
    print('epoch {}, loss {}'.format(epoch, loss))

    # 计算梯度（求导）
    grad_theta = (pred_y - y).T.dot(X) / y.size
    grad_bias = (pred_y - y).sum() / y.size

    # 更新参数
    theta -= learning_rate * grad_theta
    bias -= learning_rate * grad_bias

print('theta:\n', theta)
print('bias:\n', bias)
z = X.dot(theta) + bias
pred_y = 1 / (1 + np.exp(-z))

X1 = X[pred_y >= 0.5]
X2 = X[pred_y < 0.5]
plt.plot(X1[:, 0], X1[:, 1], 'bo')
plt.plot(X2[:, 0], X2[:, 1], 'rx')
x = np.arange(1, 3, 0.1)
y = -(theta[0] * x + bias) / theta[1]
plt.plot(x, y)
plt.show()
