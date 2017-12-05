#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable


class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(D_in, H)
        self.l2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.l1(x).clamp(min=0)
        y_pred = self.l2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).type(torch.FloatTensor)
x = Variable(x, requires_grad=False)
y = torch.randn(N, D_out).type(torch.FloatTensor)
y = Variable(y, requires_grad=False)

model = Net(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

for t in range(500):
    # forward
    y_pred = model(x)

    # loss is Variable now
    loss = criterion(y_pred, y)

    # zero the gradient before backward
    optimizer.zero_grad()
    # backward
    loss.backward()
    # update weights
    optimizer.step()

    if t % 50 == 0:
        print('Step %s Loss %s' %(t, loss.data[0]))

