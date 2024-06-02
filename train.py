import torch


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
        train_losses.append(running_loss / len(trainloader))

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(valloader))

        print(f'Epoch {epoch + 1}, Training Loss: {train_losses[-1]:.3f}, Validation Loss: {val_losses[-1]:.3f}')
        print_parameter_values(net)

    print('Finished Training')
    return train_losses, val_losses
