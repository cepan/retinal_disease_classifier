# model.py
import torch 
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights


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

def create_vit_model(num_classes, freeze_backbone=False):
    """
    Creates a Vision Transformer model with a custom classification head and optional freezing of backbone layers.
    
    Args:
        num_classes: Number of output classes for classification.
        freeze_backbone: If True, freezes the backbone layers of the ViT model.

    Returns:
        model: Modified Vision Transformer model.
    """
    # Load a pre-trained ViT model
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Ensure the classification head remains trainable
    for param in model.heads.parameters():
        param.requires_grad = True

    # Replace the classification head with a custom head for the desired number of classes
    model.heads = nn.Sequential(
        nn.Linear(model.heads[0].in_features, num_classes)
    )

    return model

class ViTWithAttention(nn.Module):
    """
    A wrapper around the Vision Transformer to extract attention weights.
    """
    def __init__(self, vit_model):
        super(ViTWithAttention, self).__init__()
        self.vit_model = vit_model
        self.attention_maps = []

        # Register hooks to capture attention from the self_attention submodule
        for block in self.vit_model.encoder.layers:
            block.self_attention.register_forward_hook(self._save_attention)

    def _save_attention(self, module, input, output):
        """
        Save attention weights during the forward pass.
        Args:
            module: The module where the hook is registered.
            input: Input to the module.
            output: Output from the module.
        """
        # Access the attention weights (input[1])
        self.attention_maps.append(input[1])  # This captures the attention weights

    def forward(self, x):
        self.attention_maps = []  # Clear previous attention maps
        logits = self.vit_model(x)
        return logits, self.attention_maps