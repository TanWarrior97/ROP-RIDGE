"""
=========================================================
  RIDGE SEGMENTATION — OPTIMIZED TRAIN + EVAL PIPELINE
  Mirrors BV/OD pipeline; ridge-specific augmentations
  Model: UNet++ / EfficientNet-B4  |  Loss: Hybrid
=========================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
import json
from pathlib import Path
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, random_split, ConcatDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ── add project root so we can import src.* ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import ROPSegmentationDataset, get_transforms
from src.model import get_segmentation_model

# =========================================================
# 0. STRICT DETERMINISM
# =========================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")

# =========================================================
# 1. RIDGE-SPECIFIC AUGMENTED TRANSFORMS
#    Ridge structures are thin, curvilinear, low-contrast.
#    We use elastic deformation + aggressive contrast to
#    force the model to generalise under clinical variability.
# =========================================================
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = (384, 384)   # same as BV / OD pipeline

def get_ridge_train_transforms():
    """Heavy augmentations tuned for thin ridge structures."""
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(shear=(-10, 10), scale=(0.90, 1.10),
                  translate_percent=(0.0, 0.05), rotate=(-20, 20),
                  border_mode=cv2.BORDER_REFLECT, p=0.6),
        # Elastic deformation: models structural variation in ridges
        A.ElasticTransform(alpha=80, sigma=8, p=0.4),
        # Photometric — ridges appear in both dim and bright fundus images
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25,
                      saturation=0.20, hue=0.10, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.11), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Coarse dropout simulates retinal artefacts / reflections
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 24),
                        hole_width_range=(16, 24), fill_value=0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_ridge_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),   # same pre-proc as train
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# =========================================================
# 2. HYBRID LOSS — ridge-specific weights
#    Ridges are very thin → high class imbalance.
#    Tversky α=0.7 β=0.3 penalises FN heavily (missed ridges).
# =========================================================
class RidgeHybridLoss(nn.Module):
    def __init__(self, pos_weight: float = 8.0):
        """
        pos_weight: ridge pixels are ~1/8 of all pixels → weight accordingly.
        """
        super().__init__()
        self.dice    = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.focal   = smp.losses.FocalLoss(smp.losses.BINARY_MODE, gamma=2.0)
        self.tversky = smp.losses.TverskyLoss(
            smp.losses.BINARY_MODE, alpha=0.7, beta=0.3, from_logits=True
        )
        pw = torch.tensor([pos_weight]).to(DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, y_pred, y_true):
        l_dice    = self.dice(y_pred, y_true)
        l_focal   = self.focal(y_pred, y_true)
        l_tversky = self.tversky(y_pred, y_true)
        l_bce     = self.bce(y_pred, y_true)
        # Tversky weighted higher — reduces missed ridges
        return l_dice + l_focal + 1.5 * l_tversky + 0.5 * l_bce

# =========================================================
# 3. DATASET BUILDER — combines Neo + RetCam (100 total)
# =========================================================
BASE = "HVDROPDB_RetCam_Neo_Segmentation"

def build_ridge_dataset(phase: str):
    """
    Merge Neo and RetCam ridge splits for maximum data.
    Returns a merged dataset with the appropriate transform.
    """
    sources = [
        (os.path.join(BASE, "HVDROPDB-RIDGE", "Neo_Ridge_images"),
         os.path.join(BASE, "HVDROPDB-RIDGE", "Neo_Ridge_masks")),
        (os.path.join(BASE, "HVDROPDB-RIDGE", "RetCam_Ridge_images"),
         os.path.join(BASE, "HVDROPDB-RIDGE", "RetCam_Ridge_masks")),
    ]
    transform = get_ridge_train_transforms() if phase == "train" else get_ridge_val_transforms()
    datasets = []
    for img_dir, msk_dir in sources:
        if os.path.isdir(img_dir) and os.path.isdir(msk_dir):
            ds = ROPSegmentationDataset(img_dir, msk_dir, transform=transform)
            datasets.append(ds)
            print(f"  [Dataset] {os.path.basename(img_dir)}: {len(ds)} samples")
        else:
            print(f"  [WARN] Directory not found: {img_dir}")
    if not datasets:
        raise RuntimeError("No ridge datasets found! Check paths.")
    return ConcatDataset(datasets)

# =========================================================
# 4. TRAINING LOOP WITH COSINE LR + WARMUP
# =========================================================
def train_ridge(epochs: int = 30, batch_size: int = 4, lr: float = 3e-4):
    """
    Full training pipeline for RIDGE segmentation.
    Saves best checkpoint to outputs/best_model_RIDGE.pth
    """
    os.makedirs("outputs", exist_ok=True)

    # ── build & split combined dataset ──────────────────────────────────────
    print("\n[PHASE 1] Building Combined Ridge Dataset ...")
    full_dataset = build_ridge_dataset(phase="train")   # returns ConcatDataset
    total_len = len(full_dataset)
    train_len = int(total_len * 0.70)
    val_len   = int(total_len * 0.15)
    test_len  = total_len - train_len - val_len
    print(f"  Total: {total_len} | Train: {train_len} | Val: {val_len} | Test: {test_len}")

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Inject val/test transform (override ConcatDataset subsets isn't trivial,
    # so we keep a separate val dataset built with val transforms)
    val_dataset  = build_ridge_dataset(phase="val")
    test_dataset = build_ridge_dataset(phase="val")
    _, val_indices, test_indices = random_split(
        list(range(total_len)), [train_len, val_len, test_len], generator=torch.Generator().manual_seed(SEED)
    )
    from torch.utils.data import Subset
    val_set_clean  = Subset(val_dataset,  val_indices.indices)
    test_set_clean = Subset(test_dataset, test_indices.indices)

    train_loader = DataLoader(train_set,     batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set_clean, batch_size=2,          shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set_clean,batch_size=2,          shuffle=False, num_workers=0, pin_memory=True)

    # ── model ──────────────────────────────────────────────────────────────
    print("\n[PHASE 2] Initialising UNet++ / EfficientNet-B4 ...")
    model = get_segmentation_model("unetplusplus", "efficientnet-b4", classes=1)

    # Resume from existing RIDGE weights if available
    ckpt_path = "outputs/best_model_RIDGE.pth"
    if os.path.exists(ckpt_path):
        print(f"  Resuming from existing checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        print("  Starting fresh (no prior RIDGE checkpoint found).")

    model = model.to(DEVICE)

    # ── optimiser & scheduler ──────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = RidgeHybridLoss(pos_weight=8.0)

    # ── training ───────────────────────────────────────────────────────────
    print(f"\n[PHASE 3] Training for {epochs} epochs ...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        # — train —
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # — validate —
        model.eval()
        val_loss = 0.0
        val_tp = val_fp = val_fn = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", leave=False):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_tp += (preds * masks).sum().item()
                val_fp += (preds * (1 - masks)).sum().item()
                val_fn += ((1 - preds) * masks).sum().item()
        val_loss /= len(val_loader)
        val_dice = (2 * val_tp) / (2 * val_tp + val_fp + val_fn + 1e-8)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_dice"].append(round(val_dice, 6))

        print(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [BEST] Best model saved (val_loss={best_val_loss:.4f})")

    # Save training history
    with open("outputs/training_history_RIDGE.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[DONE] Training complete. History saved to outputs/training_history_RIDGE.json")

    # ── final test set evaluation ──────────────────────────────────────────
    print("\n[PHASE 4] Final Test Set Evaluation ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _test_evaluation(model, test_loader, label="Test Set")

    return model

# =========================================================
# 5. DETERMINISTIC THRESHOLD SWEEP + POST-PROCESSING
#    Uses morphological thinning to preserve ridge topology
# =========================================================
def ridge_post_process(pred_mask: np.ndarray, dilation_iter: int = 1) -> np.ndarray:
    """
    1. Skeletonize → centerline
    2. Dilate back to ~3px width for clinical visibility
    """
    skeleton = skeletonize(pred_mask > 0).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined  = cv2.dilate(skeleton, kernel, iterations=dilation_iter)
    return (refined > 0).astype(np.float32)

def _test_evaluation(model, loader, label: str = "Val"):
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_masks.append(masks.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    print(f"\n  --- {label}: Threshold Sweep [0.10 to 0.90] ---")
    best_dice = 0.0
    best_t    = 0.5
    best_cm   = (0, 0, 0, 0)

    for t in np.arange(0.10, 0.91, 0.01):
        tp = fp = fn = tn = 0.0
        preds = (all_probs > t).astype(np.uint8)
        for i in range(preds.shape[0]):
            pred   = ridge_post_process(preds[i, 0])
            target = all_masks[i, 0].astype(np.float32)
            tp += np.sum(pred * target)
            fp += np.sum(pred * (1 - target))
            fn += np.sum((1 - pred) * target)
            tn += np.sum((1 - pred) * (1 - target))
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        if dice > best_dice:
            best_dice = dice
            best_t    = t
            best_cm   = (tp, fp, fn, tn)

    tp, fp, fn, tn = best_cm
    precision  = tp / (tp + fp + 1e-8)
    recall     = tp / (tp + fn + 1e-8)
    iou        = tp / (tp + fp + fn + 1e-8)
    f1         = (2 * precision * recall) / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"\n  {'='*48}")
    print(f"  RIDGE SEGMENTATION RESULTS ({label})")
    print(f"  {'='*48}")
    print(f"  Formula: Dice = 2*TP / (2*TP + FP + FN)")
    print(f"  Optimal Threshold  : {best_t:.2f}")
    print(f"  {'-'*46}")
    print(f"  Dice Score         : {best_dice:.4f}")
    print(f"  IoU  (Jaccard)     : {iou:.4f}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity        : {specificity:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  {'-'*46}")
    print(f"  Confusion Matrix (pixels):")
    print(f"    TP : {int(tp):>12,}")
    print(f"    FP : {int(fp):>12,}")
    print(f"    FN : {int(fn):>12,}")
    print(f"    TN : {int(tn):>12,}")
    print(f"  {'='*48}\n")

    return best_t, best_dice

# =========================================================
# 6. STANDALONE OPTIMIZATION EVALUATOR
#    (mirrors optimize_bv.py — load existing weights & sweep)
# =========================================================
def optimize_ridge(model_path: str = "outputs/best_model_RIDGE.pth"):
    """
    Load trained RIDGE weights and run threshold sweep on val set.
    Equivalent to optimize_bv.py but for ridge.
    """
    print("\n[OPTIMIZE] Running RIDGE Threshold Sweep ...")
    val_dataset = build_ridge_dataset(phase="val")
    total_len   = len(val_dataset)
    train_len   = int(total_len * 0.70)
    val_len     = int(total_len * 0.15)
    test_len    = total_len - train_len - val_len

    from torch.utils.data import Subset
    generator   = torch.Generator().manual_seed(SEED)
    _, val_idx, _ = random_split(list(range(total_len)),
                                  [train_len, val_len, test_len],
                                  generator=generator)
    val_loader = DataLoader(Subset(val_dataset, val_idx.indices),
                            batch_size=2, shuffle=False, num_workers=0)

    model = get_segmentation_model("unetplusplus", "efficientnet-b4", classes=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)

    best_t, best_dice = _test_evaluation(model, val_loader, label="Validation")
    print(f"[RESULT] Optimal Threshold: {best_t:.2f} | Best Dice: {best_dice:.4f}")
    return best_t, best_dice

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ridge Segmentation — Train + Eval")
    parser.add_argument("--mode",       choices=["train", "optimize", "both"], default="train",
                        help="train: full pipeline | optimize: threshold sweep only | both: train then sweep")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--model_path", type=str,   default="outputs/best_model_RIDGE.pth")
    args = parser.parse_args()

    if args.mode == "train":
        train_ridge(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.mode == "optimize":
        optimize_ridge(model_path=args.model_path)
    elif args.mode == "both":
        train_ridge(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        optimize_ridge(model_path=args.model_path)
