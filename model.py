import torch
from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input starts with dimension (B, 1, 28, 28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (B, 16, 14, 14)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (B, 16, 7, 7)
        self.fc = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        # Reshape to feed into fully connected layer.
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
        

        

        
