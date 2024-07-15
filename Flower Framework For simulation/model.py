import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(num_features // 2 * 4, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # إضافة بعد للقناة
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # تسطيح البيانات
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train(net, trainloader, optimizer, epochs, device: str):
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (features, labels) in enumerate(trainloader, 0):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print("Loss is NaN. Skipping this batch.")
                continue
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        

def test(net, testloader, device: str):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            batch_loss = criterion(outputs, labels).item()
            if torch.isnan(torch.tensor(batch_loss)):
                print("NaN detected in test loss, skipping this batch")
                continue
            loss += batch_loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
   
    return loss / len(testloader), accuracy
