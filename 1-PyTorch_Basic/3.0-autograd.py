#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).type(torch.FloatTensor)
x = Variable(x, requires_grad=False)
y = torch.randn(N, D_out).type(torch.FloatTensor)
y = Variable(y, requires_grad=False)

w1 = torch.randn(D_in, H).type(torch.FloatTensor)
w1 = Variable(w1, requires_grad=True)
w2 = torch.randn(H, D_out).type(torch.FloatTensor)
w2 = Variable(w2, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    # forward
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # loss is Variable now
    loss = (y - y_pred).pow(2).sum()

    # backward
    loss.backward()

    # update weights
    grad_w1 = w1.grad.data
    grad_w2 = w2.grad.data
    w1.data -= learning_rate*grad_w1
    w2.data -= learning_rate*grad_w2

    # zero the gradients after updating
    w1.grad.data.zero_()
    w2.grad.data.zero_()

    if t % 50 == 0:
        print('Step %s Loss %s' %(t, loss.data[0]))

