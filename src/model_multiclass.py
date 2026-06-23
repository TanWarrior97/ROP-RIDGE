import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_multiclass_segmentation_model():
    """
    Returns a unified UNet++ Architecture utilizing EfficientNet-B4
    configured for 3 output channels (Optic Disc, Blood Vessels, Ridge).
    """
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",        # Heavy duty backbone
        encoder_weights="imagenet",            # Pretrained initialization
        in_channels=3,                         # RGB Input Images
        classes=3,                             # OD, BV, Ridge Output
        activation=None                        # Soft activation applied in loss/inference natively
    )
    return model

class HybridMulticlassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Focal Loss: Heavily penalizes hard-to-classify pixels (like thin vessels and obscure ridges)
        self.focal_loss = smp.losses.FocalLoss(mode=smp.losses.MULTILABEL_MODE, alpha=0.75, gamma=2.0)
        # 2. Dice Loss: Guarantees structural/boundary overlap continuity 
        self.dice_loss = smp.losses.DiceLoss(mode=smp.losses.MULTILABEL_MODE)
        # 3. BCE Loss: Solidifies core massive structures (Optic Disc) naturally
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        loss_focal = self.focal_loss(y_pred, y_true)
        loss_dice = self.dice_loss(y_pred, y_true)
        loss_bce = self.bce_loss(y_pred, y_true)
        
        # Combined weight optimization allowing network to care about edges & density simultaneously
        return loss_focal * 0.5 + loss_dice * 0.4 + loss_bce * 0.1
