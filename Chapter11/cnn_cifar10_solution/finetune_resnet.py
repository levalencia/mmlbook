import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision import models

# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final layer to match CIFAR-10's 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# Optionally, freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Define loss and optimizer (use a smaller learning rate for fine-tuning)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

# Training loop for fine-tuning
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'resnet_cifar10.pth')
print("Fine-tuned ResNet model saved as resnet_cifar10.pth")
