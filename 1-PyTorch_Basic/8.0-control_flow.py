#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import torch
from torch.autograd import Variable


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_layer = torch.nn.Linear(D_in, H)
        self.hidden_layer = torch.nn.Linear(H, H)
        self.output_layer = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_layer(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.hidden_layer(h_relu).clamp(min=0)
        y_pred = self.output_layer(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).type(torch.FloatTensor)
x = Variable(x, requires_grad=False)
y = torch.randn(N, D_out).type(torch.FloatTensor)
y = Variable(y, requires_grad=False)

model = DynamicNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)

for t in range(1000):
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

