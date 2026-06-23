import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_classification import ROPClassificationDataset
from dataset import get_transforms
from model import get_classification_model
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # We do a basic dummy check if dataset exists. 
    # If the user hasn't provided the dataset yet, this will just gracefully tell them.
    if not os.path.exists(args.images_dir):
        print(f"Directory {args.images_dir} does not exist. Please place classification datasets to train.")
        return
        
    full_dataset = ROPClassificationDataset(args.images_dir)
    
    if len(full_dataset) == 0:
        print("No images found in dataset. Ensure class subdirectories are present.")
        return

    # Split
    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len], generator=generator)
    
    train_set.dataset.transform = get_transforms(phase="train")
    if val_len > 0:
        val_set.dataset.transform = get_transforms(phase="val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2) if val_len > 0 else None

    # Load Model
    model = get_classification_model(num_classes=2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_auc = 0.0
    output_dir = "outputs_clf"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        if val_loader:
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_probs.extend(probs[:, 1].cpu().numpy()) # P(Class 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            auc = roc_auc_score(all_labels, all_probs)
            f1 = f1_score(all_labels, all_preds, average='macro')
            acc = accuracy_score(all_labels, all_preds)
            
            print(f"Epoch {epoch+1}: Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | AUC {auc:.4f} | F1 {f1:.4f} | Acc {acc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(output_dir, "best_clf_model.pth"))
        else:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True, help="Path to classification raw images")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
