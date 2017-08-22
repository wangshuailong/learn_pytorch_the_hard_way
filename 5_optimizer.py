# /usr/bin/env python3

import torch 
import torch.utils.data as Data
import torch.nn.functional as F 
from torch.autograd import Variable
import matplotlib.pyplot as plt 

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim = 1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

torch_dataset = Data.TensorDataset(data_tensor = x, target_tensor = y)
dataloader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2
)


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

net_SGD = Net(1, 20, 1)
net_momentum = Net(1, 20, 1)
net_RMSprop = Net(1, 20, 1)
net_Adam = Net(1, 20, 1)

nets = [net_SGD, net_momentum, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr = LR)
opt_momentum = torch.optim.SGD(net_momentum.parameters(), lr = LR, momentum = 0.9)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr = LR, alpha=0.8)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr = LR, betas=(0.9, 0.99))

optimizers = [opt_SGD, opt_momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()

loss_his = [[], [], [], []]

for epoch in range(EPOCH):
    print ('EPOCH: ', epoch)

    for step, (batch_x, batch_y) in enumerate(dataloader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers, loss_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            l_his.append(loss.data[0])

print (loss_his)

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(loss_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()









