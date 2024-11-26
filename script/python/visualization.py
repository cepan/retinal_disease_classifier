import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from model import create_pretrained_model
from dataset import OcularDiseaseDataset, CropAndPadToSquare
from torch.utils.data import DataLoader
import config
from visual_util import GradCAM, visualize_heatmap,visualize_attention_map

if __name__ == '__main__':

    # Number of samples to visualize
    num_visualizations = 5
    # change as needed
    visualization_method = "grad_cam"
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_pretrained_model(num_classes=7, freeze_layers=False).to(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint)
    
    model.eval()

    # transformations 
    transform = transforms.Compose([
        transforms.ToTensor(),
        CropAndPadToSquare(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # test dataset
    test_dataset = OcularDiseaseDataset(
        img_dir=config.IMG_DIRS['test'], 
        csv_file=config.CSV_FILES['test'], 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    if visualization_method == "grad_cam":
        target_layer = model.layer4  
        grad_cam = GradCAM(model, target_layer)


    with torch.no_grad():
        total_visualized = 0
        for images, labels in test_loader:
            if total_visualized >= num_visualizations:
                break
            
            images, labels = images.to(device), labels.to(device)

            for i in range(len(images)):
                if total_visualized >= num_visualizations:
                    break
                
                image_tensor = images[i].unsqueeze(0)
                label = labels[i].item()

                # Predicted class
                outputs = model(image_tensor)
                _, predicted_class = torch.max(outputs, 1)
                predicted_label = predicted_class.item()

                if visualization_method == "grad_cam":
                    # Generate Grad-CAM heatmap
                    heatmap = grad_cam.generate_heatmap(image_tensor, predicted_label)

                    # og image for visualization
                    original_image = images[i].permute(1, 2, 0).cpu().numpy()
                    original_image = (original_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
                    original_image = original_image.astype(np.uint8)

                    # Generate overlay
                    overlay = visualize_heatmap(original_image, heatmap)

                    # Display Grad-CAM
                    plt.imshow(overlay)
                    plt.title(f"True Label: {label}, Predicted: {predicted_label}")
                    plt.axis("off")
                    plt.show()

                elif visualization_method == "attention_map":
                    # Visualize Attention Map
                    attention_map = visualize_attention_map(model, image_tensor)
                    plt.imshow(attention_map.cpu().numpy(), cmap="viridis")
                    plt.title(f"Attention Map: True Label: {label}, Predicted: {predicted_label}")
                    plt.colorbar()
                    plt.axis("off")
                    plt.show()

                total_visualized += 1
