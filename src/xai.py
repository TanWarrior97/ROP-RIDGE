import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from model import get_classification_model
from dataset import get_transforms
from albumentations.pytorch import ToTensorV2

def generate_gradcam(model, image_tensor, target_layer, target_class=1):
    model.eval()
    layer_gc = LayerGradCam(model, target_layer)
    # The baseline GradCAM algorithm requires image_tensor to require gradients
    image_tensor.requires_grad_()
    
    attributions = layer_gc.attribute(image_tensor, target_class)
    
    # Optional interpolate to image size inside visualization if needed, but captum docs suggest
    # upsampling using LayerAttribution.interpolate()
    from captum.attr import LayerAttribution
    upsampled_attr = LayerAttribution.interpolate(attributions, image_tensor.shape[2:])
    return upsampled_attr.squeeze().cpu().detach().numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = get_classification_model(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Typically resnet50 last conv layer is layer4
    target_layer = model.layer4[-1]
    
    # Load Image
    image = cv2.imread(args.image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_transforms(phase="val")
    augmented = transform(image=image_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    cam = generate_gradcam(model, image_tensor, target_layer, target_class=args.target_class)
    
    # Overlay heatmap
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # normalize
    
    # Output visualizations
    original_resized = augmented['image'].permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_resized = std * original_resized + mean
    original_resized = np.clip(original_resized, 0, 1)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    overlay = heatmap * 0.4 + original_resized
    overlay = overlay / np.max(overlay)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_resized)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')
    
    os.makedirs('xai_outputs', exist_ok=True)
    out_path = os.path.join('xai_outputs', f"gradcam_{os.path.basename(args.image_path)}")
    plt.savefig(out_path)
    print(f"Saved Grad-CAM to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--target_class", type=int, default=1, help="Class to explain")
    args = parser.parse_args()
    main(args)
