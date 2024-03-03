import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_paths, landmarks, labels, transform=None):
        self.image_paths = image_paths
        self.landmarks = landmarks
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmark = self.landmarks[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, landmark, label
    
    
# Define transformations for input images
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240, 240)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240, 240)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class FacialLandmarkNet(nn.Module):
    def __init__(self, num_classes, num_landmarks):
        super(FacialLandmarkNet, self).__init__()
        self.num_landmarks = num_landmarks
        # Define the architecture for image processing
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Adaptive average pooling layer
        # Define fully connected layers for image processing
        self.fc_image = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        # Define fully connected layers for landmarks
        self.fc_landmarks = nn.Sequential(
            nn.Linear(self.num_landmarks * 2, 512),  # Assuming each landmark has x and y coordinates
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, landmarks):
        # Process image
        x = self.features(image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_image(x)
        
        # Process landmarks
        landmarks = landmarks.view(-1, self.num_landmarks * 2)  # Flatten landmarks
        landmarks_output = self.fc_landmarks(landmarks)
        
        # Concatenate image and landmarks features
        combined_features = torch.cat((x, landmarks_output), dim=1)
        
        return combined_features