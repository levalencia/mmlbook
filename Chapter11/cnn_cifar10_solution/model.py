import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # First convolutional layer with 32 filters of size 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Second convolutional layer with 64 filters of size 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling layer with a 2x2 filter
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer with 512 units
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Assuming input size is 32x32
        
        # Final output layer with 10 units (for CIFAR-10 classes)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # First convolutional layer + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 8 * 8)
        
        # First fully connected layer + ReLU
        x = F.relu(self.fc1(x))
        
        # Output layer
        x = self.fc2(x)
        
        return x
