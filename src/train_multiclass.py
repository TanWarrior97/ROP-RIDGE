import os
import argparse
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset_multiclass import ROPMultiClassDataset, get_multiclass_transforms
from model_multiclass import get_multiclass_segmentation_model, HybridMulticlassLoss
import segmentation_models_pytorch as smp

# For automated metric generation
history = {"train_loss": [], "val_loss": [], "val_dice": [], "lr": []}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Unified Framework on {device}")
    
    # Dataset Load
    dataset = ROPMultiClassDataset(
        base_image_dir=args.base_images,
        od_mask_dir=args.od_masks,
        bv_mask_dir=args.bv_masks,
        ridge_mask_dir=args.ridge_masks,
        transform=get_multiclass_transforms("train")
    )
    
    val_dataset = ROPMultiClassDataset(
        base_image_dir=args.base_images,
        od_mask_dir=args.od_masks,
        bv_mask_dir=args.bv_masks,
        ridge_mask_dir=args.ridge_masks,
        transform=get_multiclass_transforms("val")
    )
    
    # Simple split
    dataset_size = len(dataset)
    val_size = max(1, int(0.2 * dataset_size))
    train_size = dataset_size - val_size
    train_ds, _ = random_split(dataset, [train_size, val_size])
    _, val_ds = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_multiclass_segmentation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)    
    
    # Advanced Scheduler: Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    criterion = HybridMulticlassLoss()
    metrics_calc = smp.metrics.f1_score

    os.makedirs("outputs_multiclass", exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        val_dice_total = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Metric calculations (sigmoid activation since MULTILABEL)
                probs = torch.sigmoid(outputs)
                tp, fp, fn, tn = smp.metrics.get_stats(probs, masks.long(), mode='multilabel', threshold=0.5)
                dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                val_dice_total += dice.item()

        val_loss /= len(val_loader)
        val_dice = val_dice_total / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}, LR={current_lr:.6f}")
        
        # Scheduler Step
        scheduler.step()
        
        # Metric Collection
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"outputs_multiclass/best_unified_model.pth")
            print(f"--> Saved best UNet++ model! (Val Loss: {val_loss:.4f})")

    # Serialize metrics dict for automated document generation
    with open("outputs_multiclass/training_history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_images", type=str, required=True)
    parser.add_argument("--od_masks", type=str, required=True)
    parser.add_argument("--bv_masks", type=str, required=True)
    parser.add_argument("--ridge_masks", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
