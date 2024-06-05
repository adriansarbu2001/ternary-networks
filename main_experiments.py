import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import pandas as pd

from admm_optimizer import ADMMOptimizer
from data import get_data_loaders
from models import SimpleCNN, SimpleTWN
from train import train_epoch
from evaluate import test_model
from utils import log_message


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(model, train_losses, val_losses, accuracy, cost, experiment_folder_name, lr, epoch):
    model_dir = f'results/{experiment_folder_name}/lr_{lr}'
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_path)

    # Save loss plot
    plot_path = os.path.join(model_dir, f'loss_epoch_{epoch}.png')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(plot_path)
    plt.close()

    # Save accuracy
    accuracy_path = os.path.join(model_dir, f'accuracy_epoch_{epoch}.txt')
    with open(accuracy_path, 'w') as f:
        f.write(f'{accuracy}\n')

    # Save cost
    cost_path = os.path.join(model_dir, f'cost_epoch_{epoch}.txt')
    with open(cost_path, 'w') as f:
        f.write(f'{cost}\n')


def main():
    seed = 42
    set_seed(seed)

    trainloader, valloader, testloader = get_data_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_message(f'Using device: {device}', also_print=True)

    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    # learning_rates = [0.01, 0.005]
    epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # epochs = [2, 3]

    experiments = [
        {'model_type': 'twn', 'with_batch_norm': True, 'fixed_alpha': 1.0},
        {'model_type': 'twn', 'with_batch_norm': False, 'fixed_alpha': None},
        {'model_type': 'twn', 'with_batch_norm': False, 'fixed_alpha': 1.0},
        {'model_type': 'admm', 'with_batch_norm': True, 'fixed_alpha': 1.0},
        {'model_type': 'admm', 'with_batch_norm': False, 'fixed_alpha': None},
        {'model_type': 'admm', 'with_batch_norm': False, 'fixed_alpha': 1.0},
    ]

    for experiment in experiments:
        result_accuracy = pd.DataFrame(index=epochs, columns=learning_rates)
        result_cost = pd.DataFrame(index=epochs, columns=learning_rates)

        experiment_folder_name = f'{experiment["model_type"]}_batchnorm{experiment["with_batch_norm"]}_alpha{"Unrestricted" if experiment["fixed_alpha"] is None else "Restricted"}'

        for lr in learning_rates:
            model_type = experiment["model_type"]
            with_batch_norm = experiment["with_batch_norm"]
            alpha_restricted_or_not = 'alpha not restricted' if experiment["fixed_alpha"] is None else f'alpha={experiment["fixed_alpha"]}'
            log_message(f'Starting experiment with, lr={lr}, model_type={model_type}, with_batch_norm={with_batch_norm}, {alpha_restricted_or_not}', also_print=True)

            if experiment['model_type'] == 'twn':
                net = SimpleTWN(with_batch_norm=experiment['with_batch_norm'], fixed_alpha=experiment['fixed_alpha'])
                optimizer = optim.Adam(net.parameters(), lr=lr)
            elif experiment['model_type'] == 'admm':
                net = SimpleCNN(with_batch_norm=experiment['with_batch_norm'])
                optimizer = ADMMOptimizer(net.parameters(), fixed_alpha=experiment['fixed_alpha'], base_optimizer_cls=optim.Adam, lr=lr)

            criterion = nn.CrossEntropyLoss()
            train_losses, val_losses = [], []

            for epoch in range(1, max(epochs) + 1):
                epoch_train_losses, epoch_val_losses = train_epoch(net, trainloader, valloader, criterion, optimizer, device, epoch)
                train_losses.append(epoch_train_losses)
                val_losses.append(epoch_val_losses)

                if epoch in epochs:
                    accuracy, cost = test_model(net, testloader, device)
                    save_results(net, train_losses, val_losses, accuracy, cost, experiment_folder_name, lr, epoch)

                    result_accuracy.at[epoch, lr] = accuracy
                    result_cost.at[epoch, lr] = cost

        result_accuracy.to_csv(f'results/{experiment_folder_name}/accuracy.csv')
        result_cost.to_csv(f'results/{experiment_folder_name}/cost.csv')

    log_message("")


if __name__ == "__main__":
    main()
