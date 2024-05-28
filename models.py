import torch
import torch.nn as nn
import torch.nn.functional as F


def Ternarize(tensor):
    output = torch.zeros(tensor.size(), device=tensor.device)
    delta = Delta(tensor)
    alpha = Alpha(tensor, delta)
    for i in range(tensor.size()[0]):
        pos_one = (tensor[i] > delta[i]).float()
        neg_one = -1 * (tensor[i] < -delta[i]).float()
        out = torch.add(pos_one, neg_one)
        output[i] = torch.mul(out, alpha[i])
    return output


def Alpha(tensor, delta):
    Alpha = []
    for i in range(tensor.size()[0]):
        count = 0
        abssum = 0
        absvalue = tensor[i].view(1, -1).abs()
        truth_value = absvalue > delta[i]
        count = truth_value.sum()
        abssum = (absvalue * truth_value.float()).sum()
        Alpha.append(abssum / count)
    alpha = torch.stack(Alpha)
    return alpha


def Delta(tensor):
    n = tensor[0].nelement()
    if len(tensor.size()) == 4:  # convolution layer
        delta = 0.7 * tensor.norm(1, 3).sum(2).sum(1).div(n)
    elif len(tensor.size()) == 2:  # linear layer
        delta = 0.7 * tensor.norm(1, 1).div(n)
    return delta


class TernaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(TernaryLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = Ternarize(self.weight.data)
        out = F.linear(input, self.weight, self.bias)
        return out


class TernaryConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(TernaryConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = Ternarize(self.weight.data)
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


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
