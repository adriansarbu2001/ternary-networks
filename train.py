import torch

from utils import log_message


def print_parameter_values(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            unique_values = torch.unique(param.data).cpu().numpy()
            print(f'Parameter: {name}, Unique values: {unique_values}')


def train_model(net, trainloader, valloader, criterion, optimizer, device, epochs=5):
    net.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss, val_loss = train_epoch(net, trainloader, valloader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    log_message('Finished training', also_print=True)
    return train_losses, val_losses


def train_epoch(net, trainloader, valloader, criterion, optimizer, device, epoch):
    net.to(device)
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(trainloader)

    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(valloader)

    log_message(f'Epoch {epoch}, Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}', also_print=True)
    # print_parameter_values(net)
    return train_loss, val_loss
