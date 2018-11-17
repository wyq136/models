# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import random


__author__ = 'wuyueqiu'

# 生成数据
x = np.array([i for i in range(50, 100)])
y = np.array([2.3 * i + random.randint(1, 5) for i in x])

# plt.plot(x, y, 'k.')
# plt.show()

# 定义参数 w 和 b
w = random.random()
b = 0

learning_rate = 1e-4
for epoch in range(1000):
    # 定义模型，前向计算
    pred_y = w * x + b

    # loss
    loss = 0.5 * np.square(pred_y - y).sum() / y.size
    print('epoch {}, loss {}'.format(epoch, loss))

    # 计算梯度（求导）
    grad_w = ((pred_y - y) * x).sum() / y.size
    grad_b = (pred_y - y).sum() / y.size

    # 更新参数
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

print('w: {}, b: {}'.format(w, b))
pred_y = w * x + b
plt.plot(x, y, 'k.')
plt.plot(x, pred_y, '-')
plt.show()
