import os
import torch
import numpy as np
import random
from src.dataset import ROPSegmentationDataset, get_transforms
from src.model import get_segmentation_model
from torch.utils.data import DataLoader, random_split

# Strictly enforcing deterministic execution
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

def evaluate_model(target, model_path, images_dir, masks_dir, threshold=0.5):
    print(f"\nEvaluating: {target}")
    print("="*50)

    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    # Deterministic dataset split (identical seed every run)
    full_dataset = ROPSegmentationDataset(images_dir, masks_dir, transform=None)

    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)
    val_len   = int(total_len * 0.15)
    test_len  = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Val: resize + normalize only (no augmentation)
    val_set.dataset.transform = get_transforms(phase="val")

    # Shuffle=False for deterministic loop order
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    model = get_segmentation_model("unetplusplus", "efficientnet-b4", classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    TP = FP = FN = TN = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            probs  = torch.sigmoid(logits)
            preds  = (probs > threshold).float()

            masks_f = masks.view(-1)
            preds_f = preds.view(-1)

            TP += torch.sum(preds_f * masks_f).item()
            FP += torch.sum(preds_f * (1 - masks_f)).item()
            FN += torch.sum((1 - preds_f) * masks_f).item()
            TN += torch.sum((1 - preds_f) * (1 - masks_f)).item()

    dice        = (2.0 * TP) / (2.0 * TP + FP + FN)    if (2.0*TP + FP + FN) > 0    else 0.0
    iou         = TP / (TP + FP + FN)                   if (TP + FP + FN) > 0         else 0.0
    precision   = TP / (TP + FP)                         if (TP + FP) > 0              else 0.0
    recall      = TP / (TP + FN)                         if (TP + FN) > 0              else 0.0
    specificity = TN / (TN + FP)                         if (TN + FP) > 0              else 0.0
    f1          = (2*precision*recall) / (precision+recall+1e-8)

    print(f"  Threshold          : {threshold:.2f}")
    print(f"  Formula            : Dice = 2·TP / (2·TP + FP + FN)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Dice Score         : {dice:.4f}")
    print(f"  IoU  (Jaccard)     : {iou:.4f}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity        : {specificity:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  Confusion Matrix (pixels):")
    print(f"    TP : {int(TP):>12,}")
    print(f"    FP : {int(FP):>12,}")
    print(f"    FN : {int(FN):>12,}")
    print(f"    TN : {int(TN):>12,}")
    print(f"  Final {target} Dice: {dice:.4f}")
    return dice

base_path = "HVDROPDB_RetCam_Neo_Segmentation"

# ── Optic Disc ────────────────────────────────────────────────────────────
od_images = os.path.join(base_path, "HVDROPDB-OD", "Neo_OpticDisc_images")
od_masks  = os.path.join(base_path, "HVDROPDB-OD", "Neo_OpticDisc_masks")
evaluate_model("Optic Disc (OD)", "outputs/best_model_OD.pth", od_images, od_masks)

# ── Blood Vessels ─────────────────────────────────────────────────────────
bv_images = os.path.join(base_path, "HVDROPDB-BV", "Neo_Vessels_images")
bv_masks  = os.path.join(base_path, "HVDROPDB-BV", "Neo_Vessels_masks")
evaluate_model("Blood Vessels (BV)", "outputs/best_model_BV.pth", bv_images, bv_masks)

# ── Ridge (Neo + RetCam combined → single representative eval) ────────────
# For ridge we evaluate on the Neo split (same structure as OD/BV)
ridge_images = os.path.join(base_path, "HVDROPDB-RIDGE", "Neo_Ridge_images")
ridge_masks  = os.path.join(base_path, "HVDROPDB-RIDGE", "Neo_Ridge_masks")
evaluate_model("Ridge (RIDGE)", "outputs/best_model_RIDGE.pth",
               ridge_images, ridge_masks, threshold=0.50)
