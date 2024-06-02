import torch
import torch.nn as nn
import torch.nn.functional as F


def Ternarize(tensor):
    delta = 0.7 * torch.mean(torch.abs(tensor))
    alpha = torch.mean(torch.abs(tensor[tensor.abs() >= delta]))

    tensor_tern = torch.zeros_like(tensor)
    tensor_tern[tensor.abs() >= delta] = torch.sign(tensor[tensor.abs() >= delta]) * alpha

    return tensor_tern


class TernaryConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(TernaryConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        device = input.device
        self.weight.data = Ternarize(self.weight.data.to(device))
        if self.bias is not None:
            self.bias.data = Ternarize(self.bias.data.to(device))
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class TernaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(TernaryLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        device = input.device
        self.weight.data = Ternarize(self.weight.data.to(device))
        if self.bias is not None:
            self.bias.data = Ternarize(self.bias.data.to(device))
        return F.linear(input, self.weight, self.bias)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleTWN(nn.Module):
    def __init__(self):
        super(SimpleTWN, self).__init__()
        self.conv1 = TernaryConv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = TernaryConv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = TernaryLinear(64 * 7 * 7, 128)
        self.fc2 = TernaryLinear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
