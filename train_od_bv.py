"""
=========================================================
  OD + BV SEGMENTATION -- OPTIMIZED TRAIN PIPELINE
  Trains both models back-to-back, same architecture
  as RIDGE: UNet++ / EfficientNet-B4 + Hybrid Loss
=========================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from pathlib import Path
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
import segmentation_models_pytorch as smp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import ROPSegmentationDataset, get_transforms
from src.model import get_segmentation_model

# =========================================================
# 0. DETERMINISM
# =========================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")

# =========================================================
# 1. HYBRID LOSS (same as BV original + RIDGE pipeline)
# =========================================================
class HybridLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.dice    = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.focal   = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        self.tversky = smp.losses.TverskyLoss(
            smp.losses.BINARY_MODE, alpha=0.7, beta=0.3, from_logits=True
        )
        pw = torch.tensor([pos_weight]).to(DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, y_pred, y_true):
        return (self.dice(y_pred, y_true)
                + self.focal(y_pred, y_true)
                + self.tversky(y_pred, y_true)
                + 0.5 * self.bce(y_pred, y_true))

# =========================================================
# 2. DATASET BUILDER -- merges Neo + RetCam for each target
# =========================================================
BASE = "HVDROPDB_RetCam_Neo_Segmentation"

CONFIGS = {
    "OD": {
        "sources": [
            ("HVDROPDB-OD", "Neo_OpticDisc_images",     "Neo_OpticDisc_masks"),
            ("HVDROPDB-OD", "Retcam_OpticDisc_images",  "Retcam_OpticDisc_masks"),
        ],
        "pos_weight": 3.0,   # OD is a large blob -- less imbalance
        "ckpt": "outputs/best_model_OD.pth",
    },
    "BV": {
        "sources": [
            ("HVDROPDB-BV", "Neo_Vessels_images",    "Neo_Vessels_masks"),
            ("HVDROPDB-BV", "RetCam_Vessels_images", "RetCam_Vessels_masks"),
        ],
        "pos_weight": 5.0,   # vessels are thin -- moderate imbalance
        "ckpt": "outputs/best_model_BV.pth",
    },
}

def build_dataset(target: str, phase: str):
    cfg = CONFIGS[target]
    transform = get_transforms(phase=phase)
    datasets = []
    for subdir, img_folder, msk_folder in cfg["sources"]:
        img_dir = os.path.join(BASE, subdir, img_folder)
        msk_dir = os.path.join(BASE, subdir, msk_folder)
        if os.path.isdir(img_dir) and os.path.isdir(msk_dir):
            ds = ROPSegmentationDataset(img_dir, msk_dir, transform=transform)
            datasets.append(ds)
            print(f"  [Dataset] {img_folder}: {len(ds)} samples")
        else:
            print(f"  [WARN] Not found: {img_dir}")
    if not datasets:
        raise RuntimeError(f"No data found for {target}!")
    return ConcatDataset(datasets)

# =========================================================
# 3. GENERIC TRAIN FUNCTION
# =========================================================
def train_target(target: str, epochs: int = 25, batch_size: int = 4, lr: float = 3e-4):
    os.makedirs("outputs", exist_ok=True)
    cfg = CONFIGS[target]
    ckpt_path = cfg["ckpt"]

    print(f"\n{'='*55}")
    print(f"  TRAINING: {target}  ({epochs} epochs, lr={lr})")
    print(f"{'='*55}")

    # -- datasets --
    full_train = build_dataset(target, phase="train")
    full_val   = build_dataset(target, phase="val")

    total_len = len(full_train)
    train_len = int(total_len * 0.70)
    val_len   = int(total_len * 0.15)
    test_len  = total_len - train_len - val_len
    print(f"  Total: {total_len} | Train: {train_len} | Val: {val_len} | Test: {test_len}")

    generator = torch.Generator().manual_seed(SEED)
    train_idx, val_idx, test_idx = random_split(
        list(range(total_len)), [train_len, val_len, test_len], generator=generator
    )

    train_loader = DataLoader(Subset(full_train, train_idx.indices), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(full_val,   val_idx.indices),   batch_size=2,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(Subset(full_val,   test_idx.indices),  batch_size=2,
                              shuffle=False, num_workers=0)

    # -- model --
    model = get_segmentation_model("unetplusplus", "efficientnet-b4", classes=1)
    if os.path.exists(ckpt_path):
        print(f"  Resuming from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        print(f"  Starting fresh.")
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = HybridLoss(pos_weight=cfg["pos_weight"])

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"[{target}] Ep {epoch}/{epochs} Train", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validate
        model.eval()
        val_loss = val_tp = val_fp = val_fn = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[{target}] Ep {epoch}/{epochs} Val  ", leave=False):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                out = model(images)
                val_loss += criterion(out, masks).item()
                preds = (torch.sigmoid(out) > 0.5).float()
                val_tp += (preds * masks).sum().item()
                val_fp += (preds * (1 - masks)).sum().item()
                val_fn += ((1 - preds) * masks).sum().item()
        val_loss /= len(val_loader)
        val_dice = (2 * val_tp) / (2 * val_tp + val_fp + val_fn + 1e-8)

        scheduler.step()
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_dice"].append(round(val_dice, 6))

        print(f"  Ep {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
              f"| Dice: {val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [BEST] Saved -> {ckpt_path}")

    with open(f"outputs/training_history_{target}.json", "w") as f:
        json.dump(history, f, indent=2)

    # final test sweep
    print(f"\n  --- {target} Test Set Evaluation ---")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    tp = fp = fn = tn = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = (torch.sigmoid(model(images)) > 0.5).float()
            m = masks.view(-1); p = preds.view(-1)
            tp += (p * m).sum().item()
            fp += (p * (1 - m)).sum().item()
            fn += ((1 - p) * m).sum().item()
            tn += ((1 - p) * (1 - m)).sum().item()

    dice = (2*tp) / (2*tp + fp + fn + 1e-8)
    iou  = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    print(f"  {'='*46}")
    print(f"  {target} FINAL TEST RESULTS")
    print(f"  {'='*46}")
    print(f"  Dice        : {dice:.4f}")
    print(f"  IoU         : {iou:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(f"  TP: {int(tp):,}  FP: {int(fp):,}  FN: {int(fn):,}  TN: {int(tn):,}")
    print(f"  {'='*46}")
    return dice

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  choices=["OD", "BV", "both"], default="both")
    parser.add_argument("--epochs",  type=int,   default=25)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--batch",   type=int,   default=4)
    args = parser.parse_args()

    targets = ["OD", "BV"] if args.target == "both" else [args.target]
    for t in targets:
        train_target(t, epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    print("\n[ALL DONE] Weights saved:")
    for t in targets:
        p = CONFIGS[t]["ckpt"]
        size = os.path.getsize(p) // (1024*1024) if os.path.exists(p) else 0
        print(f"  {p}  ({size} MB)")
