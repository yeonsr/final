import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default='./experiment_fashion_mnist')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args([])

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.nfe = 0  # Number of function evaluations

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_fashion_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.FashionMNIST(root='.data/fashion-mnist', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.FashionMNIST(root='.data/fashion-mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader

if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    downsampling_layers = [
        nn.Conv2d(1, 64, 3, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
    ]

    feature_layers = [ODEBlock(ODEfunc(64))] if args.network == 'odenet' else [
        nn.Conv2d(64, 64, 3, 1, 1),
        norm(64),
        nn.ReLU(inplace=True)
    ]

    fc_layers = [
        norm(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(64, 10),
    ]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_fashion_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.nepochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch + 1}/{args.nepochs}, Train Loss: {train_loss / len(train_loader):.4f}, Test Accuracy: {correct / total:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "fashion_mnist_model.pth")
    print("Model saved as fashion_mnist_model.pth")
