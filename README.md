# ROP Multi-Target Segmentation System

An advanced medical image analysis system designed for the semantic segmentation of RetCam neonatal retinal fundus scan images. The system is optimized to identify and demarcate key anatomical and pathological structures in Retinopathy of Prematurity (ROP): **Optic Disc (OD)**, **Blood Vessels (BV)**, and **demarcation Ridges (RIDGE)**.

---

## 🌟 Features

* **Specialized UNet++ Architectures:** Leverages custom UNet++ segmentation networks backed by `EfficientNet-B4` encoders for each specific target structure.
* **Intelligent Morphological Post-Processing:** Applies scikit-image skeletonization and OpenCV morphological dilation on the predicted masks to refine vessels and ridges into clean, clinically-relevant structures.
* **Cumulative Diagnostic Views:** Features a combined visual output that overlays all three segmented anatomical targets simultaneously in a unified display.
* **Interactive Glassmorphic UI:** A dark-themed Flask web application dashboard allowing doctors/researchers to upload retinal scans, execute live inference, and visualize results at original resolutions.

---

## 🎨 Segmentation Legend & Colors

| Structure | Target Description | Diagnostic Color |
| :--- | :--- | :--- |
| **Optic Disc (OD)** | Outer boundary localization of the optic nerve head. | 🔴 **Red Overlay** |
| **Blood Vessels (BV)** | Thin, skeletonized representation of the retinal vascular structure. | 🟢 **Green Overlay** |
| **Ridge (RIDGE)** | Demarcation line boundary separating vascular and avascular zones. | 🔵 **Blue Overlay** |

---

## 📁 Repository Structure

```tree
HVD_ROP_EPICS-main/
├── app.py                      # Flask web application entrypoint
├── templates/
│   └── index.html              # Responsive glassmorphic frontend dashboard
├── src/
│   ├── model.py                # UNet++ model definitions and encoder mapping
│   ├── dataset.py              # Data loading pipelines and albumentations setup
│   └── ...                     # Subproject training and evaluation scripts
├── outputs/
│   ├── best_model_OD.pth       # Specialized weights for Optic Disc
│   ├── best_model_BV.pth       # Specialized weights for Blood Vessels
│   └── best_model_RIDGE.pth    # Specialized weights for Ridge demarcation
├── requirements.txt            # System dependencies
└── run_all.ps1                 # Powershell training script for all 3 targets
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.9+** and **CUDA** (optional but highly recommended for fast GPU-accelerated inference) configured.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TanWarrior97/ROP-RIDGE.git
   cd ROP-RIDGE
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Launch the Flask Server:**
   ```bash
   python app.py
   ```
2. **Access the Dashboard:**
   Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
3. **Perform Segmentation:**
   - Upload any RetCam retinal scan image (`.png`, `.jpg`).
   - Select the target structure you want to isolate (e.g. Ridge).
   - Click **"Extract Structure"** to run inference. The UI will show the isolated target overlay alongside the cumulative visual map containing all three targets.

---

## 🛠️ Model Training

The isolated weights inside `outputs/` are generated via the specialized training scripts. To retrain the models from scratch:

```powershell
./run_all.ps1
```
This script sequentially runs:
- **Optic Disc Localization:** 50 epochs using Dice/BCE loss.
- **Vessel Skeletonization:** 50 epochs optimized for fine lines.
- **Ridge Demarcation:** 50 epochs trained on boundary demarcation datasets.
