'''
Author: Yuchen Shi
Date: 22-09-2024 13:24:50
Last Editors: Jason Shi
Contact Last Editors: Jasonycshi@outlook.com
LastEditTime: 22-09-2024 14:29:39
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from utils import *
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device.")

# Hyperparameters
num_epochs = 20
batch_size = 64
learning_rate = 0.001

# MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation in last layer for MSE
        return x


model = SimpleNN().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# One-hot encoding for labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoder.fit(np.arange(10).reshape(-1, 1))  # Fit to all possible labels

# Training loop
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.numpy().reshape(-1, 1)
        labels = one_hot_encoder.transform(labels)
        labels = torch.tensor(labels, dtype=torch.float32).to(
            device)  # Convert directly to tensor

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.numpy().reshape(-1, 1)
            labels = one_hot_encoder.transform(labels)
            labels = torch.tensor(labels, dtype=torch.float32).to(
                device)  # Convert directly to tensor

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted ==
                        labels.detach().argmax(dim=1)).sum().item()

    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(correct / total)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {
        val_loss / len(test_loader):.4f}, Validation Accuracy: {correct / total:.4f}")

# Data Visualization
# Plot the training history
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Validation Accuracy')

plt.show()


# Visualize some predictions
model.eval()
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    example_data = example_data.to(device)
    output = model(example_data)

fig = plt.figure(figsize=(12, 12))
for i in range(40):
    plt.subplot(5, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(example_data[i].cpu().numpy().squeeze(), cmap=plt.cm.binary)
    plt.xlabel(f"True: {example_targets[i]}, Pred: {
               output.argmax(dim=1)[i].item()}")
plt.show()

# Save the sample image
fig.savefig('sample_2.png')
plt.close(fig)

# Save the training history plot
plt.figure(figsize=(12, 4))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.savefig('loss_MSE.png')
plt.close()

print("Done.")
