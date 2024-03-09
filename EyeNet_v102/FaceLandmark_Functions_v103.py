import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from Net_Modules.Conv_Layers import conv,conv_110,conv_110_no_norm
from Net_Modules import Transforms as TS

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
        image = cv2.imread(img_path[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmark = self.landmarks[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
                # Convert landmark to tensor
        landmark_tensor = torch.tensor(landmark, dtype=torch.float)  # Assuming landmark is a list of coordinates

        return image, landmark_tensor, label
    
    
# Define transformations for input images
train_transform = TS.im_transform()


val_transform = TS.im_transform()


class ClassNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_110_no_norm(out_channels, out_channels),
            conv_110_no_norm(out_channels, out_channels),
            conv_110_no_norm(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x

class InitialStage(nn.Module):
    def __init__(self, num_channels, num_landmarks, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.images = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_landmarks, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.landmarks = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        images = self.images(trunk_features)
        landmarks = self.landmarks(trunk_features)
        return [images, landmarks]



class FacialLandmarkNet(nn.Module):
    def __init__(self, num_channels=128, num_landmarks=194, num_pafs=2):
        super(FacialLandmarkNet, self).__init__()
        self.num_landmarks = num_landmarks
        
        self.features = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_110( 32,  64),
            conv_110( 64, 128, stride=2),
            conv_110(128, 128),
            conv_110(128, 256, stride=2),
            conv_110(256, 256),
            conv_110(256, 512),  # conv4_2
            conv_110(512, 512, dilation=2, padding=2),
            conv_110(512, 512),
            conv_110(512, 512),
            conv_110(512, 512),
            conv_110(512, 512)   # conv5_5
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Adaptive average pooling layer
        self.landmarks = ClassNet(512(num_channels))
        self.initial_stage = InitialStage(num_channels, num_landmarks, num_pafs)

    def forward(self, x):
            face_features = self.features(x)
            face_features = self.landmarks(face_features)

            return face_features




