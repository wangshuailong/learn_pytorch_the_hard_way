#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# MNIST
train_data = dsets.MNIST(root='../data', train=True, transform=transforms.ToTensor(),
                         download=True)
test_data = dsets.MNIST(root='../data', train=False, transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32, 10)  # 28*28=>MaxPool(2)=>14*14=>MaxPool(2)=>7*7

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # view is reshape
        y = self.fc(x)
        return y


cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LEARNING_RATE)


# Train
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        outputs = cnn(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch %d/%d Iter %d/%d Loss %.4f'
                  % (epoch, EPOCHS, i, len(train_data)//BATCH_SIZE, loss.data[0]))


# Test
cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
print('Test accuracy of the model is %d %%' % (100*correct/total))