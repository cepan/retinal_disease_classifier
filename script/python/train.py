
# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from model import SimpleConvClassifier, create_pretrained_model, create_vit_model, ViTWithAttention
from dataset import OcularDiseaseDataset, CropAndPadToSquare
from train_utils import EarlyStopping, train_model, evaluate_model
import config
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.optim import Adam
import time
import numpy as np

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        CropAndPadToSquare(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees =(90,270))

    ])

    # Datasets and loaders
    train_dataset = OcularDiseaseDataset(img_dir=config.IMG_DIRS['train'], csv_file=config.CSV_FILES['train'], transform=transform)
    validation_dataset = OcularDiseaseDataset(img_dir=config.IMG_DIRS['val'], csv_file=config.CSV_FILES['val'], transform=transform)
    test_dataset = OcularDiseaseDataset(img_dir=config.IMG_DIRS['test'], csv_file=config.CSV_FILES['test'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    # criterion, optimizer
    criterion = nn.CrossEntropyLoss()
 
    # Early stopping & train
    # model = SimpleConvClassifier(num_classes=7).to(device)
    # params_to_update = [param for param in model.parameters() if param.requires_grad]
    # optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE)
    # early_stopping = EarlyStopping(patience=30, verbose=True, path=config.BEST_BASELINE_PATH)
    # train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, config.EPOCH)

    # model = create_pretrained_model(num_classes=7, freeze_layers=True).to(device)
    # params_to_update = [param for param in model.parameters() if param.requires_grad]
    # optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE)
    # early_stopping = EarlyStopping(patience=30, verbose=True, path=config.BEST_RESNET_PATH)
    # train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, config.EPOCH)


    print("==================End Training====================\n")
    print("==================End Training====================\n")


    # model = create_vit_model(num_classes=7,freeze_backbone=True)
    # model = ViTWithAttention(model).to(device)
    # params_to_update = [param for param in model.parameters() if param.requires_grad]
    # optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE)
    # early_stopping = EarlyStopping(patience=30, verbose=True, path=config.BEST_VIT_PATH)
    # train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, config.EPOCH)
    
    
    model_name = 'google/vit-base-patch16-224-in21k'
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=7).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name,do_rescale=False)
    
    # Optionally freeze backbone
    for param in model.vit.parameters():
        param.requires_grad = False

    # Prepare optimizer and criterion
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adam(params_to_update, lr=config.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=15, verbose=True, path=config.BEST_VIT_PATH)
    criterion = nn.CrossEntropyLoss()

    def preprocess_images(images):
        """Preprocess images for ViT using the processor."""
        inputs = processor(images=images, return_tensors="pt")
        return inputs

    def train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, epochs):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            print(f"Epoch [{epoch+1}/{epochs}]")
            start_time = time.time()

            for images, labels in train_loader:
                images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC
                images = np.clip(images, 0, 1)
                inputs = preprocess_images(images)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            end_time = time.time()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Time: {end_time-start_time}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.permute(0, 2, 3, 1).cpu().numpy()
                    images = np.clip(images, 0, 1)
                    inputs = preprocess_images(images)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    labels = labels.to(device)

                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(validation_loader)
            print(f'Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.2f}%')

            # Early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
    
    train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, config.EPOCH)

    

    # test set evaluation
    # model.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
    # evaluate_model(model, test_loader, criterion, device)
