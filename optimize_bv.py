import os
import torch
import numpy as np
import random
import cv2
from skimage.morphology import skeletonize
from src.dataset import ROPSegmentationDataset, get_transforms
from src.model import get_segmentation_model
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def optimize_evaluation(model_path, images_dir, masks_dir):
    full_dataset = ROPSegmentationDataset(images_dir, masks_dir, transform=None)
    
    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len
    
    generator = torch.Generator().manual_seed(seed)
    _, val_set, _ = random_split(full_dataset, [train_len, val_len, test_len], generator=generator)
    
    val_set.dataset.transform = get_transforms(phase="val")
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)
    
    model = get_segmentation_model("unetplusplus", "efficientnet-b4", classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Pre-compute probs for everything to save sweep time
    all_probs = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_masks.append(masks.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0) # [N, 1, H, W]
    all_masks = np.concatenate(all_masks, axis=0) # [N, 1, H, W]
    
    best_dice = 0.0
    best_threshold = 0.5
    best_cm = None

    kernel = np.ones((3,3), np.uint8)
    
    thresholds = np.arange(0.1, 0.95, 0.01)
    
    for t in thresholds:
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        
        preds = (all_probs > t).astype(np.uint8)
        
        for i in range(preds.shape[0]):
            pred_mask = preds[i, 0]
            true_mask = all_masks[i, 0]
            
            # Post-processing steps
            skel = skeletonize(pred_mask)
            skel_img = (skel * 255).astype(np.uint8)
            refined = cv2.dilate(skel_img, kernel, iterations=1)
            refined_bin = (refined > 127).astype(np.float32)
            
            true_bin = true_mask.astype(np.float32)
            
            TP += np.sum(refined_bin * true_bin)
            FP += np.sum(refined_bin * (1 - true_bin))
            FN += np.sum((1 - refined_bin) * true_bin)
            TN += np.sum((1 - refined_bin) * (1 - true_bin))
            
        dice = (2.0 * TP) / (2.0 * TP + FP + FN) if (2.0*TP + FP + FN) > 0 else 0.0
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = t
            best_cm = (TP, FP, FN, TN)
            
    print(f"--- DET_OPT RUN END ---")
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Optimal Dice: {best_dice:.4f}")
    print(f"TP: {int(best_cm[0])}, FP: {int(best_cm[1])}, FN: {int(best_cm[2])}, TN: {int(best_cm[3])}")

base_path = "HVDROPDB_RetCam_Neo_Segmentation"
bv_images = os.path.join(base_path, "HVDROPDB-BV", "Neo_Vessels_images")
bv_masks = os.path.join(base_path, "HVDROPDB-BV", "Neo_Vessels_masks")
optimize_evaluation("outputs/best_model_BV.pth", bv_images, bv_masks)
