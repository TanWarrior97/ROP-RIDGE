import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import ROPSegmentationDataset, get_transforms
from model import get_segmentation_model
from tqdm import tqdm
import segmentation_models_pytorch as smp

def create_dataloaders(images_dir, masks_dir, batch_size=8, split_ratio=(0.7, 0.15, 0.15)):
    # Simple validation split setup. In real 3-fold CV we would use sklearn KFold. 
    # Providing a base test loop matching Phase 1 structure.
    full_dataset = ROPSegmentationDataset(images_dir, masks_dir, transform=None)
    
    total_len = len(full_dataset)
    train_len = int(total_len * split_ratio[0])
    val_len = int(total_len * split_ratio[1])
    test_len = total_len - train_len - val_len
    
    # Needs generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len], generator=generator)
    
    # Inject transforms post-split
    train_set.dataset.transform = get_transforms(phase="train")
    if len(val_set) > 0:
        val_set.dataset.transform = get_transforms(phase="val")
    if len(test_set) > 0:
        test_set.dataset.transform = get_transforms(phase="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2) if val_len > 0 else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2) if test_len > 0 else None
    
    return train_loader, val_loader, test_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Loaders
    train_loader, val_loader, test_loader = create_dataloaders(args.images_dir, args.masks_dir, batch_size=args.batch_size)
    
    # Model
    model = get_segmentation_model(model_name=args.model_name, encoder_name=args.encoder, in_channels=args.in_channels, classes=args.classes)
    model = model.to(device)
    
    # Losses & Optimizer
    # Using BCEWithLogits as standard out of UNet typically lacks final sigmoid
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss composition
            loss = criterion(outputs, masks) + 0.5 * bce_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks) + 0.5 * bce_loss(outputs, masks)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_{args.target}.pth"))
        else:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(output_dir, f"latest_model_{args.target}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="OD", help="Target structure element (OD, BV, RIDGE)")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to raw images")
    parser.add_argument("--masks_dir", type=str, required=True, help="Path to masks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_name", type=str, default="unet")
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--classes", type=int, default=1)
    
    args = parser.parse_args()
    train(args)
