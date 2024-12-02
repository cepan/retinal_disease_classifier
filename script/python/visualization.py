import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from model import create_pretrained_model, create_vit_model,ViTWithAttention
from dataset import OcularDiseaseDataset, CropAndPadToSquare
from torch.utils.data import DataLoader
import config
from visual_util import GradCAM, visualize_heatmap,visualize_attention
from transformers import ViTForImageClassification, ViTImageProcessor




if __name__ == '__main__':

    # Number of samples to visualize
    num_visualizations = 3
    # change as needed
    visualization_method = "grad_cam"

    label_mapping = {
    0: "DR",    # Diabetic Retinopathy
    1: "MH",    # Macular Hole
    2: "ODC",   # Optic Disc Cupping
    3: "TSLN",  # Tesselated Fundus
    4: "DN",    # Drusen
    5: "MYA",   # Myopia
    6: "ARMD"   # Age-Related Macular Degeneration
    }

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
        
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if visualization_method == "grad_cam":
        model = create_pretrained_model(num_classes=7, freeze_layers=False).to(device)
        checkpoint = torch.load(config.BEST_RESNET_PATH, weights_only=True)
        target_layer = model.layer4  
        grad_cam = GradCAM(model, target_layer)
        # Load state_dict if checkpoint includes more metadata
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    else:
        model_name = 'google/vit-base-patch16-224-in21k'
        processor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=7,attn_implementation="eager")
        checkpoint = torch.load(config.BEST_VIT_PATH, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

    
    model = model.to(device)
    model.eval()



    with torch.no_grad():
        total_visualized = 0
        for images, labels in test_loader:
            # Check if enough visualizations have been generated
            if total_visualized >= num_visualizations:
                break
            
            images, labels = images.to(device), labels.to(device)


            for i in range(len(images)):
                if total_visualized >= num_visualizations:
                    break

                image_tensor = images[i].unsqueeze(0)
                label = labels[i].item()
                

                if visualization_method == "grad_cam":


                    # Predicted class
                    outputs = model(image_tensor)
                    if isinstance(outputs, tuple):  # Handle tuple output (logits, attentions)
                        logits, attentions = outputs
                    else:
                        logits = outputs  # Handle single output case

                    _, predicted_class = torch.max(logits, 1)
                    predicted_label = predicted_class.item()

                    if label  == 6:
                        

                        # Grad-CAM Visualization
                        heatmap = grad_cam.generate_heatmap(image_tensor, predicted_label)

                        original_image = images[i].permute(1, 2, 0).cpu().numpy()
                        original_image = (original_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
                        original_image = original_image.astype(np.uint8)

                        overlay = visualize_heatmap(original_image, heatmap)

                        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                        axs[0].imshow(original_image)
                        axs[0].axis('off')
                        axs[0].set_title(f"Original Image: {label_mapping[label]}")

                        axs[1].imshow(overlay)
                        axs[1].axis('off')
                        axs[1].set_title(f"Overlay (Predicted: {label_mapping[predicted_label]})")

                        plt.tight_layout()
                        plt.subplots_adjust(wspace=0.15)
                        plt.savefig(f'../../visualizations/grad_cam/ARMD/figure_{total_visualized}.png')
                        # plt.show()
                        plt.close()
                                                
                        total_visualized += 1


                elif visualization_method == "attention_map":

                    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(-1, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(-1, 1, 1)

                    denormalized_image = image_tensor * std + mean
                    denormalized_image = torch.clamp(denormalized_image, 0, 1)

                    inputs = processor(images=denormalized_image, return_tensors="pt")
                    inputs = {key: val.to(device) for key, val in inputs.items()}

                    outputs = model(**inputs, output_attentions=True)
                    logits = outputs.logits
                    predicted_class = logits.argmax(dim=-1).item() 
                    
                    if label  == 6:
                        # Getting the attentions
                        attentions = outputs.attentions 
                        average_attention = attentions[-1].mean(dim=1)
                        cls_attention = average_attention[0, 0, 1:].reshape(14, 14)
                        resized_attention = cv2.resize(cls_attention.cpu().numpy(), (224, 224))
                        resized_attention = (resized_attention - resized_attention.min()) / (resized_attention.max() - resized_attention.min())

                        original_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
                        original_image = (original_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
                        original_image = original_image.astype(np.uint8)

                        # Create overlay
                        heatmap = cv2.applyColorMap((resized_attention * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

                        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                        axs[0].imshow(original_image)
                        axs[0].axis('off')
                        axs[0].set_title(f"Original Image: {label_mapping[label]}")

                        axs[1].imshow(overlay)
                        axs[1].axis('off')
                        axs[1].set_title(f"Overlay (Predicted: {label_mapping[predicted_class]})")

                        plt.tight_layout()
                        plt.subplots_adjust(wspace=0.15)
                        plt.savefig(f'../../visualizations/attention map/MYA/figure_{total_visualized}.png')
                        plt.close()
                    

                        # Increment visualization counter
                        total_visualized += 1
