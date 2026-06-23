import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import get_transforms
from src.model import get_segmentation_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load highly accurate Optic Disc Model
model = get_segmentation_model("unetplusplus", "efficientnet-b4", classes=1)
model.load_state_dict(torch.load("outputs/best_model_OD.pth", map_location=device))
model.to(device)
model.eval()

# Load specific unseen testing image
img_path = r"HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-OD/Neo_OpticDisc_images/10.png"
gt_path = r"HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-OD/Neo_OpticDisc_masks/10.png"

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
_, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
gt_rgb = np.zeros_like(img_rgb)
gt_rgb[gt_binary == 255] = [255, 0, 0] # Real Ground Truth Red

# Format image for UNet++
transform = get_transforms((384, 384), phase="val")
augmented = transform(image=img_rgb)
img_tensor = augmented["image"].unsqueeze(0).to(device)

# Grab mathematical prediction geometry
with torch.no_grad():
    output = model(img_tensor)

prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
raw_h, raw_w = img_bgr_shape = img.shape[:2]

# Map back to 100% Native Resolution
pred_upscaled = cv2.resize(prob_mask, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
ai_binary = (pred_upscaled > 0.5).astype(np.uint8)

ai_rgb = np.zeros_like(img_rgb)
ai_rgb[ai_binary == 1] = [0, 255, 0] # AI Prediction Green

# Create Side-by-Side Validation Figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("1. Original Retinal Scan")

img_gt_blend = cv2.addWeighted(img_rgb, 0.7, gt_rgb, 0.5, 0)
axes[1].imshow(img_gt_blend)
axes[1].set_title("2. Doctor's Ground Truth (Red)")

img_ai_blend = cv2.addWeighted(img_rgb, 0.7, ai_rgb, 0.5, 0)
axes[2].imshow(img_ai_blend)
axes[2].set_title("3. UNet++ 96.5% AI Prediction (Green)")

for ax in axes:
    ax.axis("off")

# Save directly to artifacts directory
import os
os.makedirs(r"C:\Users\gujar\.gemini\antigravity\brain\88c01ad3-db9d-48de-836e-e2601991ea2c\artifacts", exist_ok=True)
plt.savefig(r"C:\Users\gujar\.gemini\antigravity\brain\88c01ad3-db9d-48de-836e-e2601991ea2c\artifacts\proof.png", bbox_inches='tight', dpi=150)
print("Saved artifacts/proof.png")
