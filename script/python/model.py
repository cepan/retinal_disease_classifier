# model.py
import torch 
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16


class SimpleConvClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 128 * 128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def create_pretrained_model(num_classes=7, freeze_layers=True):
    
    model = models.resnet50(weights="IMAGENET1K_V2")
    
    if freeze_layers:
        # Freeze initial layers
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    # Replace the fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def create_vit_model(num_classes=7, pretrained=True,freeze_backbone = True):
    # Load the pretrained ViT model
    model = vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)

    if freeze_backbone:
        # Freeze all layers except the classifier head
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model