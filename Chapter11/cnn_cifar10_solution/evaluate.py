import torch
import torchvision
import torchvision.transforms as transforms
from model import CustomCNN  # Import the model

# Define test data transformations
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load the saved model
model = CustomCNN()
model.load_state_dict(torch.load('cnn_cifar10.pth'))
model.eval()  # Set model to evaluation mode

# Evaluate accuracy
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the CIFAR-10 test set: {accuracy:.2f}%')
