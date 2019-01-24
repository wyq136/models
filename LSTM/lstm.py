# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np


__author__ = 'wuyueqiu'


class LSTMCell:
    """
    某一个时刻的 LSTM 计算单元
    """
    def __init__(self, in_size, hidden_size):
        self.in_size = in_size
        self.hidden_size = hidden_size
        # 输入、遗忘、输出门参数
        self.w_ii = np.random.normal(0, 0.1, (in_size, hidden_size))
        self.w_hi = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_i = np.random.normal(0, 0.1, (1, hidden_size))
        self.w_if = np.random.normal(0, 0.1, (in_size, hidden_size))
        self.w_hf = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_f = np.random.normal(0, 0.1, (1, hidden_size))
        self.w_io = np.random.normal(0, 0.1, (in_size, hidden_size))
        self.w_ho = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_o = np.random.normal(0, 0.1, (1, hidden_size))

        # 当前时刻前向计算参数
        self.w_ig = np.random.normal(0, 0.1, (in_size, hidden_size))
        self.w_hg = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_g = np.random.normal(0, 0.1, (1, hidden_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, out_sigmoid):
        return out_sigmoid * (1 - out_sigmoid)

    def tanh(self, x):
        # (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return 2 * self.sigmoid(2 * x) - 1

    def tanh_grad(self, out_tanh):
        return 1 - out_tanh * out_tanh

    def forward(self, x, h, c):
        # 输入、遗忘、输出门
        i = x.dot(self.w_ii) + h.dot(self.w_hi) + self.b_i
        input_gate = self.sigmoid(i)
        f = x.dot(self.w_if) + h.dot(self.w_hf) + self.b_f
        forget_gate = self.sigmoid(f)
        o = x.dot(self.w_io) + h.dot(self.w_ho) + self.b_o
        output_gate = self.sigmoid(o)

        # 前馈网络
        g = x.dot(self.w_ig) + h.dot(self.w_hg) + self.b_g
        g_tanh = self.tanh(g)

        # 新的长期记忆单元
        cell = forget_gate * c + input_gate * g_tanh

        # 输出结果
        cell_tanh = self.tanh(cell)
        h_output = output_gate * cell_tanh

        step_output = (input_gate, forget_gate, output_gate, g_tanh, cell_tanh)
        return h_output, cell, step_output

    def backward(self, grad, x, h, c, step_output):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if h.ndim == 1:
            h = np.expand_dims(h, axis=0)
        if c.ndim == 1:
            c = np.expand_dims(c, axis=0)
        batch_size = x.shape[0]
        b_ones = np.ones((1, batch_size))

        input_gate, forget_gate, output_gate, g_tanh, cell_tanh = step_output

        # 输出门梯度
        grad_output_gate = grad * cell_tanh
        grad_o = grad_output_gate * self.sigmoid_grad(output_gate)
        self.grad_w_io = x.T.dot(grad_o)
        self.grad_w_ho = h.T.dot(grad_o)
        self.grad_b_o = b_ones.dot(grad_o)

        grad_cell_tanh = grad * output_gate
        grad_cell = grad_cell_tanh * self.tanh_grad(cell_tanh)
        # 遗忘门梯度
        grad_forget_gate = grad_cell * c
        grad_f = grad_forget_gate * self.sigmoid_grad(forget_gate)
        self.grad_w_if = x.T.dot(grad_f)
        self.grad_w_hf = h.T.dot(grad_f)
        self.grad_b_f = b_ones.dot(grad_f)

        # 输入门梯度
        grad_input_gate = grad_cell * g_tanh
        grad_i = grad_input_gate * self.sigmoid_grad(input_gate)
        self.grad_w_ii = x.T.dot(grad_i)
        self.grad_w_hi = h.T.dot(grad_i)
        self.grad_b_i = b_ones.dot(grad_i)

        # 前馈网络梯度
        grad_g_tanh = grad_cell * input_gate
        grad_g = grad_g_tanh * self.tanh_grad(g_tanh)
        self.grad_w_ig = x.T.dot(grad_g)
        self.grad_w_hg = h.T.dot(grad_g)
        self.grad_b_g = b_ones.dot(grad_g)

        # 上一个时刻隐藏层（这一层的隐藏层输入）梯度
        grad_h_in = grad_g.dot(self.w_hg) + grad_i.dot(self.w_hi) + grad_f.dot(self.w_hf) + grad_o.dot(self.w_ho)
        return grad_h_in


class LSTM:
    def __init__(self, in_size, hidden_size):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.h_state_list = []
        self.c_state_list = []
        self.step_output_list = []
        self.lstm_cell = LSTMCell(in_size, hidden_size)

    def forward(self, x):
        self.h_state_list = []
        self.c_state_list = []
        self.step_output_list = []
        # x 输入的第一维是 batch size，第二维是序列长度，第三维是向量维度
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        self.x = x
        batch_size = x.shape[0]
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        for i in range(x.shape[1]):
            self.h_state_list.append(h)
            self.c_state_list.append(c)
            h, c, step_output = self.lstm_cell.forward(x[:, i, :], h, c)
            self.step_output_list.append(step_output)
        self.h_output = h
        return self.h_output

    def zero_grad(self):
        # 梯度清零
        self.grad_w_i = np.zeros((4 * self.in_size, self.hidden_size))
        self.grad_w_h = np.zeros((4 * self.hidden_size, self.hidden_size))
        self.grad_b = np.zeros((4, self.hidden_size))

    def backward(self, grad):
        self.zero_grad()
        for i in range(len(self.h_state_list) - 1, -1, -1):
            x = self.x[:, i, :]
            h = self.h_state_list[i]
            c = self.c_state_list[i]
            step_output = self.step_output_list[i]
            grad = self.lstm_cell.backward(grad, x, h, c, step_output)

            # 将所有时刻的梯度累加
            self.grad_w_i += np.concatenate([self.lstm_cell.grad_w_ii,
                                             self.lstm_cell.grad_w_if,
                                             self.lstm_cell.grad_w_io,
                                             self.lstm_cell.grad_w_ig], axis=0)

            self.grad_w_h += np.concatenate([self.lstm_cell.grad_w_hi,
                                             self.lstm_cell.grad_w_hf,
                                             self.lstm_cell.grad_w_ho,
                                             self.lstm_cell.grad_w_hg], axis=0)

            self.grad_b += np.concatenate([self.lstm_cell.grad_b_i,
                                           self.lstm_cell.grad_b_f,
                                           self.lstm_cell.grad_b_o,
                                           self.lstm_cell.grad_b_g], axis=0)

        return grad

    def update_weight(self, lr):
        self.lstm_cell.w_ii -= lr * self.grad_w_i[: self.in_size, ]
        self.lstm_cell.w_hi -= lr * self.grad_w_h[: self.hidden_size, ]
        self.lstm_cell.b_i -= lr * self.grad_b[0, ]

        self.lstm_cell.w_if -= lr * self.grad_w_i[self.in_size: self.in_size * 2, ]
        self.lstm_cell.w_hf -= lr * self.grad_w_h[self.hidden_size: self.hidden_size * 2, ]
        self.lstm_cell.b_f -= lr * self.grad_b[1, ]

        self.lstm_cell.w_io -= lr * self.grad_w_i[self.in_size * 2: self.in_size * 3, ]
        self.lstm_cell.w_ho -= lr * self.grad_w_h[self.hidden_size * 2: self.hidden_size * 3, ]
        self.lstm_cell.b_o -= lr * self.grad_b[2, ]

        self.lstm_cell.w_ig -= lr * self.grad_w_i[self.in_size * 3:, ]
        self.lstm_cell.w_hg -= lr * self.grad_w_h[self.hidden_size * 3:, ]
        self.lstm_cell.b_g -= lr * self.grad_b[3:, ]

    def gradient_check(self):
        pass
