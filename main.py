import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np

from admm_optimizer import ADMMOptimizer
from data import get_data_loaders
from models import SimpleCNN, SimpleTWN
from train import train_model
from evaluate import test_model
from utils import plot_losses


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(model_type="cnn"):
    seed = 42
    set_seed(seed)

    trainloader, valloader, testloader = get_data_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    lr = 0.01
    print(f'Learning rate: {lr}')

    net = SimpleCNN()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if model_type == "cnn":
        net = SimpleCNN(with_batch_norm=True)
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif model_type == "twn":
        net = SimpleTWN(with_batch_norm=False)
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif model_type == "admm":
        net = SimpleCNN(with_batch_norm=False)
        optimizer = ADMMOptimizer(net.parameters(), base_optimizer_cls=optim.Adam, lr=lr)

    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = train_model(net, trainloader, valloader, criterion, optimizer, device, epochs=50)
    torch.save(net.state_dict(), f'simple_{model_type}.pth')
    plot_losses(train_losses, val_losses)

    net.load_state_dict(torch.load(f'simple_{model_type}.pth'))
    test_model(net, testloader, device)


if __name__ == "__main__":
    main(model_type="twn")
