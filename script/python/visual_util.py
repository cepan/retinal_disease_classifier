import torch
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt




class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Attach hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hooks on the target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, inputs, class_idx):
        # Enable gradient calculation explicitly
        with torch.set_grad_enabled(True):
            # Forward 
            outputs = self.model(inputs)
            self.model.zero_grad()

            # Backward 
            target = outputs[:, class_idx]
            target.backward()

            # Ensure gradients are captured
            if self.gradients is None:
                raise ValueError("Gradients are not captured. Check if hooks are correctly registered.")

            # Compute Grad-CAM heatmap
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # Global average pooling
            heatmap = torch.zeros_like(self.activations[0, 0])
            for i in range(pooled_gradients.shape[0]):
                heatmap += pooled_gradients[i] * self.activations[0, i]

            heatmap = torch.clamp(heatmap, min=0).detach().cpu().numpy()
            heatmap /= np.max(heatmap)
            return heatmap



def visualize_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay



def visualize_attention(model, dataloader, device):
    """
    Visualizes the attention map for a batch of images.
    Args:
        model: Trained Vision Transformer with attention extraction.
        dataloader: DataLoader for test images.
        device: Device to run the model on.
    """
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, attentions = model(images)
            if attentions is not None:
                # Select one sample and visualize its attention
                attention_map = attentions[-1][0].cpu().numpy()  # Take last layer's attention
                # Plot attention map here
                print("Attention map shape:", attention_map.shape)
            else:
                print("No attention maps available.")
            break


