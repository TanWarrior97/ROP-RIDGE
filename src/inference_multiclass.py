import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from model_multiclass import get_multiclass_segmentation_model
from dataset_multiclass import get_multiclass_transforms

def run_multiclass_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model (UNet++ EfficientNet-B4)
    model = get_multiclass_segmentation_model()
    if not os.path.exists(args.model_path):
        print(f"Error: Unified model not found at {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Process Input
    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = get_multiclass_transforms("val")
    augmented = transform(image=img_rgb)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        
    # Get probabilities between 0 and 1
    probs = torch.sigmoid(output).squeeze().cpu().numpy() # Shape [3, H, W]
    
    # Thresholds
    od_pred = (probs[0] > 0.5).astype(np.uint8)
    bv_pred = (probs[1] > 0.5).astype(np.uint8)
    ridge_pred = (probs[2] > 0.5).astype(np.uint8)
    
    # Post-Processing: Morphological Thinning / Skeletonization for Blood Vessels
    # Skeletonize requires binary boolean input
    skeleton_bv = skeletonize(bv_pred > 0)
    bv_thinned = skeleton_bv.astype(np.uint8)
    
    # Reverse Augmentation norm to get original background
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    view_img = augmented["image"].permute(1, 2, 0).cpu().numpy()
    view_img = std * view_img + mean
    view_img = np.clip(view_img, 0, 1)
    
    base_img_rgb = np.uint8(view_img * 255)
    
    # Generate Clean Composite Mask
    # Red(OD), Green(Thin BV), Blue(Ridge)
    composite = np.zeros_like(base_img_rgb)
    composite[od_pred == 1] = [255, 0, 0]          # Red OD
    composite[bv_thinned == 1] = [0, 255, 0]       # Green Thinned Vessels
    composite[ridge_pred == 1] = [0, 0, 255]       # Blue Ridge
    
    # Create Blended Overlays for visual interpretation
    # We only blend where predictions exist
    mask_indices = np.any(composite > 0, axis=-1)
    blended = base_img_rgb.copy()
    blended[mask_indices] = cv2.addWeighted(
        base_img_rgb[mask_indices], 0.3, 
        composite[mask_indices], 0.7, 0
    )
    
    # Heatmaps
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.mean(probs, axis=0)), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Save outputs visually
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(base_img_rgb)
    axes[0].set_title("Input Retinal Image")
    
    axes[1].imshow(composite)
    axes[1].set_title("RGB Multiclass Segments")
    
    axes[2].imshow(blended)
    axes[2].set_title("Clinical Overlay (Thinned BV)")
    
    axes[3].imshow(heatmap_colored)
    axes[3].set_title("Multi-Target Attention Heatmap")
    
    for ax in axes:
        ax.axis("off")
        
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, bbox_inches='tight', dpi=150)
    print(f"Saved inference mapping to: {args.output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="outputs_multiclass/best_unified_model.pth")
    parser.add_argument("--output_png", type=str, default="composite_prediction.png")
    args = parser.parse_args()
    run_multiclass_inference(args)
