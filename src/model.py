import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_segmentation_model(model_name="unet", encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1):
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,     
            in_channels=in_channels,                  
            classes=classes,                      
        )
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    return model

# Baseline Classification Models (Phase 2)
from torchvision.models import resnet50, ResNet50_Weights

def get_classification_model(num_classes=2):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Replace final linear layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

class MultimodalFusionROP(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Branch 1: Raw Image Processor (ResNet-50)
        self.image_branch = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_branch.fc = nn.Identity() # Strip final layer, gives 2048 vector
        
        # Branch 2: Segmentation Processor (Processing OD, BV, Ridge Masks)
        # Using a shallow CNN designed to extract structural features from a 3-channel stacked mask
        self.mask_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, masks):
        img_features = self.image_branch(img)
        mask_features = self.mask_branch(masks)
        combined = torch.cat((img_features, mask_features), dim=1)
        out = self.fusion(combined)
        return out
