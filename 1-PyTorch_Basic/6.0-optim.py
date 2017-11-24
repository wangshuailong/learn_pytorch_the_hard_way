#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).type(torch.FloatTensor)
x = Variable(x, requires_grad=False)
y = torch.randn(N, D_out).type(torch.FloatTensor)
y = Variable(y, requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)


loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

for t in range(500):
    # forward
    y_pred = model(x)

    # loss is Variable now
    loss = loss_fn(y_pred, y)

    # zero the gradient before backward
    optimizer.zero_grad()

    # backward
    loss.backward()

    # update weights
    optimizer.step()

    if t % 50 == 0:
        print('Step %s Loss %s' %(t, loss.data[0]))

