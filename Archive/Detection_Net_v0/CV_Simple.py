import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
from PIL import Image

# Custom transformation to ensure output is a PIL image
class ToPILImageWithConvert:
    def __call__(self, img):
        return Image.fromarray(img)

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations for input images
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a simple CNN model inspired by EfficientNet
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(
            InvertedResidualBlock(32, 16, expansion_factor=1, stride=1),
            InvertedResidualBlock(16, 24, expansion_factor=6, stride=2),
            InvertedResidualBlock(24, 32, expansion_factor=6, stride=2),
            InvertedResidualBlock(32, 64, expansion_factor=6, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Define Inverted Residual Block
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = expansion_factor * in_channels
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(hidden_dim, hidden_dim, stride=stride),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.stride = stride

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and residual.shape[1] == x.shape[1]:
            residual += x
        return residual

# Depthwise Separable Convolution Module
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Initialize the CNN model
model = CNNModel(num_classes=10)  # Example with 10 classes

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

train_image_paths,train_labels = ['20240202_133511.jpg'],[0]
val_image_paths,val_labels = ['20240202_133511.jpg'],[0]

train_dataset = CustomDataset(train_image_paths, train_labels, transform=train_transform)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=val_transform)

# Define DataLoader for training and validation sets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        # print(inputs)
        inputs = inputs.to(device)
        # print(labels)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    # Validate the model
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print statistics
    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%')
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

print('Training complete.')


# Example usage
image_path = '20240202_133511.jpg'
# prediction = model(transform(cv2.imread(image_path)).unsqueeze(0))
# print("Prediction:", prediction.argmax(dim=1))
