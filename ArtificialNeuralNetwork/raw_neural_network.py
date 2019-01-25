# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
import utils


__author__ = 'wuyueqiu'

# 生成数据
X, y = utils.gen_data()

input_size = 2
hidden_size = 5
output_size = 1

# 生成参数theta
theta1 = np.random.rand(input_size + 1, hidden_size)
theta2 = np.random.rand(hidden_size, output_size)

# 添加偏差项
ones = np.ones((X.shape[0], 1))
X = np.concatenate((X, ones), axis=1)

learning_rate = 1e-1
for epoch in range(10001):
    # 定义模型，前向计算（这里的隐藏层没有添加偏差项，也可以在每层隐藏层都加上偏差项）
    z2 = X.dot(theta1)
    a2 = 1 / (1 + np.exp(-z2))
    z3 = a2.dot(theta2)
    pred_y = 1 / (1 + np.exp(-z3))

    # loss
    loss = - (y * np.log(pred_y) + (1 - y) * np.log(1 - pred_y)).mean()
    print('epoch {}, loss {}'.format(epoch, loss))

    # 输出训练时的准确率
    if epoch % 100 == 0:
        pred_label = pred_y >= 0.5
        true_label = y >= 0.5
        diff = pred_label == true_label
        accuracy = diff.mean()
        print('accuracy {}'.format(accuracy))

    # 使用链式求导计算梯度（和反向传播是一致的）
    grad_z3 = pred_y - y
    grad_theta2 = a2.T.dot(grad_z3)
    grad_a2 = grad_z3.dot(theta2.T)
    grad_z2 = grad_a2 * a2 * (1 - a2)
    grad_theta1 = X.T.dot(grad_z2)

    # 更新参数
    theta2 -= learning_rate * grad_theta2
    theta1 -= learning_rate * grad_theta1

print('theta1:\n', theta1)
print('theta2:\n', theta2)

z2 = X.dot(theta1)
a2 = 1 / (1 + np.exp(-z2))
z3 = a2.dot(theta2)
pred_y = 1 / (1 + np.exp(-z3))

utils.show_data(X[:, :2], pred_y)
