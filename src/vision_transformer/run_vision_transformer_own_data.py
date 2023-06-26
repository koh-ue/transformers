#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torchvision
import torch.nn as nn
from einops import repeat
import torch.optim as optim
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

sys.path.append(".")

from src.vision_transformer.vit_model import *

if __name__ == "__main__":
    batch_size = 100

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    print(train_set)
    sys.exit()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = ViT(
        image_size=32,
        patch_size=4,
        n_classes=10,
        dim=256,
        depth=3,
        n_heads=4,
        mlp_dim = 256
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    epochs = 10
    for epoch in range(0, epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_test_loss = 0
        epoch_test_acc = 0

        net.train()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()/len(train_loader)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_train_acc += acc/len(train_loader)

            del inputs
            del outputs
            del loss

        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()/len(test_loader)
                test_acc = (outputs.argmax(dim=1) == labels).float().mean()
                epoch_test_acc += test_acc/len(test_loader)

        print(f'Epoch {epoch+1} : train acc. {epoch_train_acc:.2f} train loss {epoch_train_loss:.2f}')
        print(f'Epoch {epoch+1} : test acc. {epoch_test_acc:.2f} test loss {epoch_test_loss:.2f}')