import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import get_segmentation_model
from dataset import get_transforms
from albumentations.pytorch import ToTensorV2

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    
    # 1. Load Model Weight
    model = get_segmentation_model(model_name="unet", encoder_name="resnet50", in_channels=3, classes=1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Process Input Image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Could not load image: {args.image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Validation augmentations (clean normalize)
    transform = get_transforms(phase="val")
    augmented = transform(image=img_rgb)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    # 3. Model Inference (Forward pass)
    with torch.no_grad():
        output = model(img_tensor)
        
    # Standard Sigmoid to turn raw logits -> Probabilities between 0.0 and 1.0
    probs = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Convert back to standard boolean mask mapping (predict > 0.5)
    pred_mask = (probs > 0.5).astype(np.uint8)

    # 4. Process Optional Ground Truth Mask
    gt_mask = None
    if args.mask_path and os.path.exists(args.mask_path):
        gt = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is not None:
             # Just resize gt to display roughly next to prediction
             gt_mask = cv2.resize(gt, (probs.shape[1], probs.shape[0]))
             _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
             gt_mask = gt_mask / 255.0

    # 5. Reverse normalization to view the original image properly
    # (Matches training normalization params exactly)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    view_img = augmented["image"].permute(1, 2, 0).cpu().numpy()
    view_img = std * view_img + mean
    view_img = np.clip(view_img, 0, 1)

    # 6. Build the Visual Plot
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))
    axes[0].imshow(view_img)
    axes[0].set_title("Input (Normalized Crop)")
    axes[0].axis("off")
    
    col = 1
    if gt_mask is not None:
        axes[col].imshow(gt_mask, cmap='gray')
        axes[col].set_title("Ground Truth Mask")
        axes[col].axis("off")
        col += 1
        
    axes[col].imshow(pred_mask, cmap='magma')
    axes[col].set_title("AI Prediction Mask")
    axes[col].axis("off")
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, bbox_inches='tight')
    print(f"Saved visual evaluation to: {args.output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_png", type=str, default="evaluation.png")
    args = parser.parse_args()
    run_inference(args)
