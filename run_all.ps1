Write-Host "========================================="
Write-Host "   Starting Isolated Multi-Target Train"
Write-Host "========================================="

Write-Host "`n➤ Training Optic Disc Locator (UNet++ EfficientNet-B4 | 50 Epochs)"
python src/train.py --images "HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-OD/Neo_OpticDisc_images" --masks "HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-OD/Neo_OpticDisc_masks" --output "outputs/best_model_OD.pth" --epochs 50 --batch_size 2

Write-Host "`n➤ Training Blood Vessel Skeletonizer (UNet++ EfficientNet-B4 | 50 Epochs)"
python src/train.py --images "HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-BV/Neo_Vessels_images" --masks "HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-BV/Neo_Vessels_masks" --output "outputs/best_model_BV.pth" --epochs 50 --batch_size 2

Write-Host "`n➤ Training Ridge Demarcator (UNet++ EfficientNet-B4 | 50 Epochs)"
python src/train.py --images "HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-RIDGE/Neo_Ridge_images" --masks "HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-RIDGE/Neo_Ridge_masks" --output "outputs/best_model_RIDGE.pth" --epochs 50 --batch_size 2

Write-Host "`n✅ All Isolated Architectures Successfully Generated!"
