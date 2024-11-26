
# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from model import SimpleConvClassifier, create_pretrained_model, create_vit_model
from dataset import OcularDiseaseDataset, CropAndPadToSquare
from train_utils import EarlyStopping, train_model, evaluate_model
import config


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

    model = create_pretrained_model(num_classes=7, freeze_layers=True).to(device)
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=30, verbose=True, path=config.BEST_RESNET_PATH)
    train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, config.EPOCH)


    print("==================End Training====================\n")
    print("==================End Training====================\n")


    model = create_vit_model(num_classes=7).to(device)
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=30, verbose=True, path=config.BEST_VIT_PATH)
    train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, config.EPOCH)

    

    # test set evaluation
    # model.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
    # evaluate_model(model, test_loader, criterion, device)
