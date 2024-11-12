
# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from model import SimpleConvClassifier, create_pretrained_model
from dataset import OcularDiseaseDataset, CropAndPadToSquare
from train_utils import EarlyStopping, train_model, evaluate_model
import config


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        CropAndPadToSquare(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    # Datasets and loaders
    train_dataset = OcularDiseaseDataset(img_dir=config.IMG_DIRS['train'], csv_file=config.CSV_FILES['train'], transform=transform)
    validation_dataset = OcularDiseaseDataset(img_dir=config.IMG_DIRS['val'], csv_file=config.CSV_FILES['val'], transform=transform)
    test_dataset = OcularDiseaseDataset(img_dir=config.IMG_DIRS['test'], csv_file=config.CSV_FILES['test'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    # Model, criterion, optimizer
    model = SimpleConvClassifier(num_classes=7).to(device)
    model = create_pretrained_model(num_classes=7, freeze_layers=True).to(device)
    criterion = nn.CrossEntropyLoss()
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True, path=config.BEST_MODEL_PATH)

    train_model(model, train_loader, validation_loader, criterion, optimizer, early_stopping, device, config.EPOCH)


    # model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    # evaluate_model(model, test_loader, criterion, device)
