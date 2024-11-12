# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from torchvision import transforms
import torch.nn.functional as F



def crop_and_pad_to_square(image):
        
    # Convert to grayscale
    gray = torch.mean(image, dim=0, keepdim=True)

    # Use a threshold to create binary mask
    binary_mask = (gray > 0.1).float()


    # Find the non-zero elements
    non_zero_indices = torch.nonzero(binary_mask[0])


    if non_zero_indices.size(0) == 0: 
        return image

    # Get the bounding box
    top_left = torch.min(non_zero_indices, dim=0)[0]
    bottom_right = torch.max(non_zero_indices, dim=0)[0]

    cropped_image = image[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # Calculate padding to square the image
    delta_w = bottom_right[1] - top_left[1]
    delta_h = bottom_right[0] - top_left[0]
    padding = (delta_h - delta_w) // 2

    # Pad and return the square image
    square_image = F.pad(cropped_image, (padding, padding, padding, padding), mode='constant', value=0)
    return square_image


class CropAndPadToSquare:
    def __call__(self, image):
        # Define the crop_and_pad_to_square function here as it’s part of this class.
        return crop_and_pad_to_square(image)

class OcularDiseaseDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        
        other_conditions = self.df.iloc[idx][['DR', 'MH', 'ODC', 'TSLN', 'DN', 'MYA', 'ARMD']].values
        label = torch.tensor([*other_conditions], dtype=torch.float32)
        multi_class_label = torch.argmax(label).item()
        
        return img, multi_class_label
