#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################
# You may have to try more than 1 time to see a good result  #
# Should be related with gradient vanishing or exploding #
# Could be solved by some ways like better initialization #
# But I won't do here to keep it a minimal implementation #
##############################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

DATA_SIZE = 60
EPOCHS = 10000

x_input_size = 60
z_input_size = 60
hidden_size = 128

x_base = np.linspace(-2, 2, DATA_SIZE)


def get_x_data():
    a = np.random.uniform(0.8, 1.2, size=DATA_SIZE)
    x = a*np.power(x_base, 2)
    x = x[np.newaxis, :]
    return x


def get_z_data():
    z = np.random.rand(1, DATA_SIZE)
    return z


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.map1(x))
        y = self.map2(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        y = F.sigmoid(x)
        return y


D = Discriminator(input_size=x_input_size, hidden_size=hidden_size, output_size=1)
G = Generator(input_size=z_input_size, hidden_size=hidden_size, output_size=x_input_size)

opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)

plt.ion()
for epoch in range(EPOCHS):
    # train discriminator net
    # discriminate processed random (fake) data from generator
    z = get_z_data()
    z = torch.from_numpy(z).float()
    z = Variable(z)
    gz = G(z)
    dz = D(gz)

    # discriminate data from real data
    x = get_x_data()
    real_data = np.squeeze(x)
    x = torch.from_numpy(x).float()
    x = Variable(x)
    dx = D(x)

    D_loss = -torch.mean((torch.log(dx) + torch.log(1.-dz)))
    G_loss = torch.mean(torch.log(1.-dz))

    opt_D.zero_grad()
    D_loss.backward(retain_variables=True)      # retain_variables for reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if epoch % 20 == 0:
        plt.cla()
        generated_data = np.squeeze(gz.data.numpy().reshape(-1, 1))
        plt.plot(x_base, real_data, color='blue', label='Real Data')
        plt.plot(x_base, generated_data, color='red', label='Generated Data')
        plt.xlim((-2.2, 2.2))
        plt.ylim((-1, 5))
        plt.legend(loc='upper right', fontsize=12)
        plt.pause(0.001)

plt.show()
plt.ioff()