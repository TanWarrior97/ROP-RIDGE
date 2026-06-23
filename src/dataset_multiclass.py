import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ROPMultiClassDataset(Dataset):
    def __init__(self, base_image_dir, od_mask_dir, bv_mask_dir, ridge_mask_dir, transform=None):
        self.base_image_dir = base_image_dir
        self.od_mask_dir = od_mask_dir
        self.bv_mask_dir = bv_mask_dir
        self.ridge_mask_dir = ridge_mask_dir
        self.transform = transform
        
        self.images = [f for f in os.listdir(base_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load Base Image
        img_path = os.path.join(self.base_image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Function to safely load masks (Returns black mask if missing)
        def load_mask(mask_dir):
            if mask_dir is None:
                return np.zeros(image.shape[:2], dtype=np.uint8)
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                return mask
            return np.zeros(image.shape[:2], dtype=np.uint8)
            
        # Stack channels [H, W, 3] -> Optic Disc (0), Blood Vessels (1), Ridge (2)
        od_mask = load_mask(self.od_mask_dir)
        bv_mask = load_mask(self.bv_mask_dir)
        ridge_mask = load_mask(self.ridge_mask_dir)
        
        stacked_masks = np.stack([od_mask, bv_mask, ridge_mask], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=stacked_masks)
            image = augmented['image']
            stacked_masks = augmented['mask']
            
        # Ensure mask is [C, H, W] FloatTensor for SMP loss compatibility
        stacked_masks = stacked_masks.permute(2, 0, 1).float()

        return image, stacked_masks

def get_multiclass_transforms(phase):
    if phase == "train":
        return A.Compose([
            A.Resize(384, 384),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.8),
            A.CLAHE(p=0.5), # Enhance local contrast for tiny ridge patterns
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2), # Injections
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
