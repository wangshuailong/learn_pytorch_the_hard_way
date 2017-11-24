#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).type(torch.FloatTensor)
y = torch.randn(N, D_out).type(torch.FloatTensor)

w1 = torch.randn(D_in, H).type(torch.FloatTensor)
w2 = torch.randn(H, D_out).type(torch.FloatTensor)

learning_rate = 1e-6

for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    loss = (y - y_pred).pow(2).sum()

    grad_y_pred = 2*(y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

    if t % 50 == 0:
        print('Step %s Loss %s' %(t, loss))

