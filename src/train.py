import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import ROPSegmentationDataset, get_transforms
from model import get_segmentation_model
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Isolated Model on {device}")
    
    dataset = ROPSegmentationDataset(
        images_dir=args.images, 
        masks_dir=args.masks, 
        transform=get_transforms((384, 384), "train")
    )
    val_dataset = ROPSegmentationDataset(
        images_dir=args.images, 
        masks_dir=args.masks, 
        transform=get_transforms((384, 384), "val")
    )
    
    dataset_size = len(dataset)
    val_size = max(1, int(0.2 * dataset_size))
    train_size = dataset_size - val_size
    train_ds, _ = random_split(dataset, [train_size, val_size])
    _, val_ds = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # UNet++ with EfficientNet-B4 for isolated target extraction
    model = get_segmentation_model(
        model_name="unetplusplus",
        encoder_name="efficientnet-b4",
        in_channels=3,
        classes=1
    ).to(device)

    # Isolated Loss: Weighted BCE + Dice Loss
    bce = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE)
    
    def criterion(pred, target):
        return bce(pred, target) * 0.5 + dice(pred, target) * 0.5

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        val_dice_total = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                tp, fp, fn, tn = smp.metrics.get_stats(probs, masks.long(), mode='binary', threshold=0.5)
                val_dice_total += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()

        val_loss /= len(val_loader)
        val_dice = val_dice_total / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValDice={val_dice:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"--> Saved best isolated model to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_single(args)
