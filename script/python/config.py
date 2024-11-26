

BATCH_SIZE = 16
EPOCH = 100
LEARNING_RATE = 0.001
BEST_VIT_PATH = '../../save_models/best_vit_model.pth'
BEST_BASELINE_PATH = '../../save_models/best_baseline_model.pth'
BEST_RESNET_PATH = '../../save_models/best_resnet_model.pth'

IMG_DIRS = {
    'train': '../../data/RFMiD/img/Train',
    'val': '../../data/RFMiD/img/Validation',
    'test': '../../data/RFMiD/img/Test'
}
CSV_FILES = {
    'train': '../../data/RFMiD/labels/Filtered_Train.csv',
    'val': '../../data/RFMiD/labels/Filtered_Validation.csv',
    'test': '../../data/RFMiD/labels/Filtered_Test.csv'
}
