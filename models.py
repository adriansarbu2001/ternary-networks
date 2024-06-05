import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ternarize


class TernaryConv2d(nn.Conv2d):
    def __init__(self, fixed_alpha=None, *args, **kwargs):
        super(TernaryConv2d, self).__init__(*args, **kwargs)
        self.fixed_alpha = fixed_alpha

    def forward(self, input):
        device = input.device
        if self.fixed_alpha is None:
            alpha = torch.mean(torch.abs(self.weight.data[self.weight.data.abs() >= 0.7 * torch.mean(torch.abs(self.weight.data))]))
        else:
            alpha = self.fixed_alpha
        self.weight.data = ternarize(self.weight.data.to(device), alpha=alpha)
        if self.bias is not None:
            self.bias.data = ternarize(self.bias.data.to(device), alpha=alpha)
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class TernaryLinear(nn.Linear):
    def __init__(self, fixed_alpha=None, *args, **kwargs):
        super(TernaryLinear, self).__init__(*args, **kwargs)
        self.fixed_alpha = fixed_alpha

    def forward(self, input):
        device = input.device
        if self.fixed_alpha is None:
            alpha = torch.mean(torch.abs(self.weight.data[self.weight.data.abs() >= 0.7 * torch.mean(torch.abs(self.weight.data))]))
        else:
            alpha = self.fixed_alpha
        self.weight.data = ternarize(self.weight.data.to(device), alpha=alpha)
        if self.bias is not None:
            self.bias.data = ternarize(self.bias.data.to(device), alpha=alpha)
        return F.linear(input, self.weight, self.bias)


class SimpleCNN(nn.Module):
    def __init__(self, with_batch_norm=False):
        super(SimpleCNN, self).__init__()
        self.with_batch_norm = with_batch_norm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.with_batch_norm:
            x = self.bn_conv1(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = self.conv2(x)
        if self.with_batch_norm:
            x = self.bn_conv2(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def set_batch_norm(self, with_batch_norm):
        self.with_batch_norm = with_batch_norm


class SimpleTWN(nn.Module):
    def __init__(self, with_batch_norm=False, fixed_alpha=None):
        super(SimpleTWN, self).__init__()
        self.with_batch_norm = with_batch_norm
        self.conv1 = TernaryConv2d(fixed_alpha, 1, 32, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = TernaryConv2d(fixed_alpha, 32, 64, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = TernaryLinear(fixed_alpha, 1024, 512)
        self.fc2 = TernaryLinear(fixed_alpha, 512, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.with_batch_norm:
            x = self.bn_conv1(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = self.conv2(x)
        if self.with_batch_norm:
            x = self.bn_conv2(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def set_batch_norm(self, with_batch_norm):
        self.with_batch_norm = with_batch_norm
