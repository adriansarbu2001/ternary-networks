import matplotlib.pyplot as plt
import torch
import datetime


def ternarize(tensor, alpha):
    delta = 0.7 * torch.mean(torch.abs(tensor))
    tensor_tern = torch.zeros_like(tensor)
    tensor_tern[tensor.abs() >= delta] = torch.sign(tensor[tensor.abs() >= delta]) * alpha
    return tensor_tern


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


def log_message(message, log_file='experiment_log.txt', also_print=False):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(f'[{timestamp}] {message}\n')
    if also_print:
        print(f'[{timestamp}] {message}')
