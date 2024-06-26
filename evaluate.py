import torch


def test_model(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    nonzero_weights = sum((param != 0).sum().item() for param in net.parameters())

    return accuracy, nonzero_weights
