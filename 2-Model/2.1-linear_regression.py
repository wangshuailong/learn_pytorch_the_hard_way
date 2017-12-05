# /usr/bin/env python3

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.1*torch.randn(x.size())

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x


net = Net(n_feature = 1, n_hidden = 10, n_output = 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
criterion = torch.nn.MSELoss()

plt.ion()

for t in range(300):
    prediction = net(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
        plt.text(0.5, 0, "Loss = %.4f" % loss.data[0], fontdict={'size': 10, 'color': 'blue'})
        plt.pause(0.1)

plt.ioff()
plt.show()



