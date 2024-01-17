import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os, ssl
import io
import pickle
from my_cifar import CIFAR10
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

if __name__ == '__main__':
    # GPU를 사용할 경우 mac user
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # CIFAR-10 데이터셋 불러오기 및 전처리
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root='./data', train=True,
                                           download=False, transform=transform, extract=False)
#    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                              download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=3, worker_init_fn=CIFAR10.worker_init_fn)

    # 신경망 정의
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, 2)
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x, 2)
            x = x.view(-1, 32 * 8 * 8)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()
    model.to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    
    # 모델 학습
    for epoch in range(1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
        # Compute the average loss and accuracy
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc:.2f}%")
        
    #print(model)
    #print(model.parameters)
    #print(model.state_dict)
    # with open('data.pt', 'wb') as f:
    #     pickle.dump(model.state_dict(), f, pickle.HIGHEST_PROTOCOL)

    torch.save(model.state_dict(), './checkpoint_test.pth')    
    print("Finished Training")

# model = models.resnet18

