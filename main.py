import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
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

    net = SimpleCNN()
    if model_type == "cnn":
        net = SimpleCNN()
    elif model_type == "twn":
        net = SimpleTWN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    train_losses, val_losses = train_model(net, trainloader, valloader, criterion, optimizer, device)
    test_model(net, testloader, device)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main(model_type="twn")


"""
Using device: cuda:0
Epoch 1, Training Loss: 0.520, Validation Loss: 0.145
Epoch 2, Training Loss: 0.117, Validation Loss: 0.101
Epoch 3, Training Loss: 0.077, Validation Loss: 0.101
Epoch 4, Training Loss: 0.050, Validation Loss: 0.093
Epoch 5, Training Loss: 0.037, Validation Loss: 0.073
Finished Training
Accuracy of the network on the 2000 test images: 98.00%
"""

"""

"""