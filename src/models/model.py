import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    """A simple convolutional neural network model for image classification

    Args:
    num_classes (int, optional): Number of output classes. Defaults to 10.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Input starts with dimension (B, 1, 28, 28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # (B, 16, 14, 14)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # (B, 16, 7, 7)
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        """Pass input through the model

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 28, 28)

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        # Conversion to float was apparently needed to run the code
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        # Reshape to feed into fully connected layer.
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
