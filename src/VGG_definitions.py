import torch
from torchvision import transforms, models
from torch.utils.data import Dataset
from PIL import Image
import os

# model_definitions.py

import torch.nn as nn
import torch.nn.functional as F

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, dataframe, directory, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 2]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx, 3]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# Attention modules
class SoftAttention(nn.Module):
    def __init__(self, in_channels, multiheads, concat_with_x=False, aggregate=False):
        super(SoftAttention, self).__init__()
        self.channels = in_channels
        self.multiheads = multiheads
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x
        self.conv = nn.Conv2d(in_channels, multiheads, kernel_size=3, padding=1)
        
    def forward(self, x):
        attention = self.conv(x)
        attention = F.relu(attention)
        batch, _, height, width = attention.size()
        attention = attention.view(batch, self.multiheads, -1)
        attention = F.softmax(attention, dim=-1)
        attention = attention.view(batch, self.multiheads, height, width)
        
        if not self.aggregate_channels:
            attention = attention.unsqueeze(2)
            x = x.unsqueeze(1)
            output = x * attention
            output = output.view(batch, -1, height, width)
        else:
            attention = attention.sum(dim=1, keepdim=True)
            output = x * attention
        
        if self.concat_input_with_scaled:
            output = torch.cat([output, x], dim=1)
        return output, attention.squeeze(1)

# Main model
class VGG16WithAttention(nn.Module):
    def __init__(self):
        super(VGG16WithAttention, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        
        self.features_1 = nn.Sequential(*list(vgg16.features.children())[:24])
        self.attention = SoftAttention(in_channels=512, multiheads=1, aggregate=True)
        self.features_2 = nn.Sequential(*list(vgg16.features.children())[24:])
        self.avgpool = vgg16.avgpool
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1)
        )
    
    def forward(self, x):
        x = self.features_1(x)
        x, attention_weights = self.attention(x)
        x = self.features_2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)

# Define transformations
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])