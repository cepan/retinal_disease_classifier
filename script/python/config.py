

BATCH_SIZE = 16
EPOCH = 100
LEARNING_RATE = 0.001
BEST_MODEL_PATH = '../../save_models/best_model.pth'
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
