import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_data_loaders(batch_size=64, train_size=10000, val_size=5000, test_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(trainset))[:train_size]
    val_indices = torch.randperm(len(trainset))[:val_size]
    test_indices = torch.randperm(len(testset))[:test_size]

    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(trainset, val_indices)
    test_subset = Subset(testset, test_indices)

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader
