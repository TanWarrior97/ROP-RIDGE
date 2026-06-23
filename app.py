from flask import Flask, request, jsonify, render_template
import torch
import cv2
import numpy as np
import base64
import os
from skimage.morphology import skeletonize
from src.dataset import get_transforms
from src.model import get_segmentation_model

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_specific_model(target):
    # Mapping exact isolated weights dynamically
    if target == "OD":
        path = "outputs/best_model_OD.pth"
    elif target == "BV":
        path = "outputs/best_model_BV.pth"
    elif target == "RIDGE":
        path = "outputs/best_model_RIDGE.pth"
    else:
        return None
        
    if not os.path.exists(path):
        return None
        
    print(f"Loading Specialized UNet++ EfficientNet-B4 for {target}...")
    model = get_segmentation_model(
        model_name="unetplusplus",
        encoder_name="efficientnet-b4",
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

@app.route('/')
def home():
    return render_template('index.html')

MODELS = {}

def get_cached_model(target):
    if target in MODELS:
        return MODELS[target]
    model = load_specific_model(target)
    if model is not None:
        MODELS[target] = model
    return model

def process_target_prediction(model, target, img_rgb, img_tensor, raw_w, raw_h):
    with torch.no_grad():
        output = model(img_tensor)

    prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_upscaled = cv2.resize(prob_mask, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (pred_upscaled > 0.5).astype(np.uint8)

    target_color = [0, 0, 0]
    
    if not np.any(binary_mask):
        return np.zeros((raw_h, raw_w), dtype=np.uint8), target_color
    if target == "OD":
        final_mask = binary_mask
        target_color = [255, 0, 0] # Red
    elif target == "BV":
        skeleton_bv = skeletonize(binary_mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(skeleton_bv, kernel, iterations=1)
        target_color = [0, 255, 0] # Green
    elif target == "RIDGE":
        skeleton_ridge  = skeletonize(binary_mask > 0).astype(np.uint8)
        kernel_ridge    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask      = cv2.dilate(skeleton_ridge, kernel_ridge, iterations=1)
        target_color    = [0, 0, 255]  # Blue

    return final_mask, target_color

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'target' not in request.form:
        return jsonify({'success': False, 'error': 'Missing upload or target.'})

    target = request.form['target']
    isolated_model = get_cached_model(target)
    
    if isolated_model is None:
        return jsonify({'success': False, 'error': f'Model weights for {target} not found. Ensure specific isolated training is completed.'})

    try:
        file = request.files['image']
        npimg = np.fromfile(file, np.uint8)
        img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        raw_h, raw_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        transform = get_transforms((384, 384), phase="val")
        augmented = transform(image=img_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

        # 1. Compute Isolated Map
        iso_mask, iso_color = process_target_prediction(isolated_model, target, img_rgb, img_tensor, raw_w, raw_h)
        iso_blended = img_rgb.copy()
        iso_overlay = np.zeros_like(img_rgb)
        iso_overlay[iso_mask == 1] = iso_color
        idx = iso_mask == 1
        if np.any(idx):
            iso_blended[idx] = cv2.addWeighted(img_rgb[idx], 0.3, iso_overlay[idx], 0.7, 0)
        
        _, iso_buffer = cv2.imencode('.png', cv2.cvtColor(iso_blended, cv2.COLOR_RGB2BGR))
        iso_b64 = base64.b64encode(iso_buffer).decode('utf-8')

        # 2. Compute Cumulative Map
        cum_blended = img_rgb.copy()
        cum_overlay = np.zeros_like(img_rgb)
        cum_mask_any = np.zeros((raw_h, raw_w), dtype=bool)

        for t in ["OD", "BV", "RIDGE"]:
            m = get_cached_model(t)
            if m is not None:
                m_mask, m_color = process_target_prediction(m, t, img_rgb, img_tensor, raw_w, raw_h)
                cum_overlay[m_mask == 1] = m_color
                cum_mask_any = cum_mask_any | (m_mask == 1)

        if np.any(cum_mask_any):
            cum_blended[cum_mask_any] = cv2.addWeighted(img_rgb[cum_mask_any], 0.3, cum_overlay[cum_mask_any], 0.7, 0)
        
        _, cum_buffer = cv2.imencode('.png', cv2.cvtColor(cum_blended, cv2.COLOR_RGB2BGR))
        cum_b64 = base64.b64encode(cum_buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'prediction_base64': iso_b64,
            'cumulative_base64': cum_b64
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
