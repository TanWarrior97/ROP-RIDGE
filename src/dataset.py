import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ROPSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Valid image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        self.images_list = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
        # Ensure mapping with masks
        # Assuming mask names either perfectly match or have some known standard (_mask, etc)
        # Note: the user mentioned masks are in directories like `Neo_OpticDisc_masks`.
        # We will assume identical filenames or at least easily mappable ones for now.
        
    def __len__(self):
        return len(self.images_list)
        
    def __getitem__(self, idx):
        img_name = self.images_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Assume mask name same as img_name or ends with something similar
        # For HVDROPDB, generally names correspond. Let's try direct map:
        # Also handle potential png mask for jpg image
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, base_name + ".jpg") # fallback
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.masks_dir, img_name)
                
        # Read image and mask
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found/readable: {mask_path}")
            
        # Binarize mask
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Albumentations with ToTensorV2 converts to FloatTensor [C, H, W] for image
        # If no transform or transform doesn't convert:
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        else:
            mask = mask.float()
            
        return image, mask.unsqueeze(0) # Mask shape [1, H, W]

def get_transforms(img_size=(384, 384), phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
