import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ROPClassificationDataset(Dataset):
    def __init__(self, images_dir, labels_dict=None, transform=None):
        """
        images_dir: path to directory with classification images
        labels_dict: dictionary mapping filename to label (e.g., {'1.jpg': 1, '2.jpg': 0})
                     If None, tries to parse from directory structure (e.g. Normal/ vs ROP/)
        """
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        if labels_dict is not None:
            for f in os.listdir(images_dir):
                if f.lower().endswith(valid_extensions):
                    self.samples.append((f, labels_dict[f]))
        else:
            # Assume subdirectories point to class directories
            classes = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
            if len(classes) == 0:
                raise ValueError("No classes found. Provide labels_dict or use class subdirectories.")
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            for cls_name in classes:
                cls_dir = os.path.join(images_dir, cls_name)
                for f in os.listdir(cls_dir):
                    if f.lower().endswith(valid_extensions):
                        self.samples.append((os.path.join(cls_name, f), class_to_idx[cls_name]))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        return image, torch.tensor(label, dtype=torch.long)

class MultimodalROPDataset(Dataset):
    def __init__(self, images_dir, seg_model_paths, labels_dict=None, transform=None):
        # We assume seg_model_paths is a dict to multiple loaded segmentation models 
        # e.g., {'OD': model1, 'BV': model2, 'Ridge': model3}
        # In a real multimodal, we pass the image through these models and concatenate channels
        # Or you can pre-generate masks and load them here.
        pass # To be implemented when datasets are available
