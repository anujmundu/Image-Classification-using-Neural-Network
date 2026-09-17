import base64
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import build_model
from src.data_loader import get_data_loaders

# Grad-CAM via torchcam (already in requirements)
from torchcam.methods import GradCAM

# SHAP (DeepExplainer)
import shap

# Optional timm for ViT attention maps
import timm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path: Path, num_classes: int = None, backbone: str = 'resnet101'):
    state_dict = torch.load(model_path, map_location=DEVICE)
    if num_classes is None:
        classes_file = model_path.parent / 'classes.json'
        if classes_file.exists():
            try:
                with open(classes_file, 'r') as f:
                    num_classes = len(json.load(f))
            except Exception:
                pass
        if num_classes is None:
            for key in ['fc.weight', 'classifier.1.weight', 'classifier.6.weight', 'head.weight']:
                if key in state_dict:
                    num_classes = state_dict[key].shape[0]
                    break
        if num_classes is None:
            num_classes = 2

    model = build_model(num_classes=num_classes, backbone=backbone)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# Preprocess image (common for CNN/ViT)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def image_to_tensor(image_path: Path):
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0).to(DEVICE)

def get_gradcam(model, tensor, pred_idx, backbone: str, image_path: Path):
    """Return visual explainability heatmap overlay bytes (base64 PNG).
    Supports Grad-CAM for CNNs (ResNet, EfficientNet) and gradient saliency maps
    for ViT, MLP-Mixer, and CapsNet.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        orig_img = Image.open(image_path).convert('RGB').resize((224, 224))
        orig_arr = np.array(orig_img, dtype=np.float32) / 255.0

        cam_mask = None

        # 1. Try TorchCAM for standard CNNs
        if backbone in ['resnet101', 'efficientnet_b4', 'vgg16', 'alexnet']:
            try:
                from torchcam.methods import GradCAM
                target_layer = 'features.8' if backbone == 'efficientnet_b4' else ('layer4' if backbone == 'resnet101' else None)
                with GradCAM(model, target_layer=target_layer) as cam_extractor:
                    model.zero_grad()
                    out = model(tensor)
                    cams = cam_extractor(pred_idx, out)
                    if cams and len(cams) > 0:
                        cam_mask = cams[0].squeeze().detach().cpu().numpy()
            except Exception:
                cam_mask = None

        # 2. Universal Gradient Saliency Fallback (Works on ViT, MLP-Mixer, CapsNet, and CNNs)
        if cam_mask is None:
            tensor_req = tensor.clone().detach().requires_grad_(True)
            out = model(tensor_req)
            score = out[0, pred_idx]
            model.zero_grad()
            score.backward()
            grads = tensor_req.grad.detach().abs().squeeze().cpu().numpy()  # (3, H, W)
            cam_mask = np.mean(grads, axis=0)  # (H, W)

        # Normalize heatmap to [0, 1]
        cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min() + 1e-8)
        
        # Resize mask to 224x224 if needed
        if cam_mask.shape != (224, 224):
            from PIL import Image as PilImage
            mask_pil = PilImage.fromarray((cam_mask * 255).astype(np.uint8)).resize((224, 224), PilImage.Resampling.BILINEAR)
            cam_mask = np.array(mask_pil, dtype=np.float32) / 255.0

        # Apply colormap (jet)
        cmap = plt.get_cmap('jet')
        heatmap = cmap(cam_mask)[:, :, :3]  # (224, 224, 3)

        # Blend original image and heatmap
        blended = (0.55 * orig_arr + 0.45 * heatmap)
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Warning: Visual heatmap generation failed: {e}", flush=True)
        return ''

def get_shap_explanation(model, tensor, train_loader):
    """Compute SHAP values for a single input using a small background set.
    Returns base64 PNG of the SHAP plot.
    """
    try:
        background = []
        for i, (x, _) in enumerate(train_loader):
            background.append(x.to(DEVICE))
            if i >= 2:
                break
        background = torch.cat(background, dim=0)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(tensor)
        shap_img = shap.image_plot(shap_values, tensor.cpu().numpy(), show=False)
        buf = io.BytesIO()
        shap_img[0].figure.savefig(buf, format='png')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return ''

def predict(image_path: Path, model_path: Path, class_names: list = None, backbone: str = 'resnet101', include_shap: bool = False, loaded_model = None):
    if class_names is None:
        classes_file = model_path.parent / 'classes.json'
        if classes_file.exists():
            try:
                with open(classes_file, 'r') as f:
                    class_names = json.load(f)
            except Exception:
                class_names = ['cats', 'dogs']
        else:
            class_names = ['cats', 'dogs']

    num_classes = len(class_names)
    model = loaded_model if loaded_model is not None else load_model(model_path, num_classes, backbone)
    tensor = image_to_tensor(image_path)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
        label = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
        confidence = float(probs[pred_idx])

    # Top-5 rankings for multi-class classification
    top_k = min(5, len(class_names))
    top_indices = probs.argsort()[-top_k:][::-1]
    top5 = [{'label': class_names[i] if i < len(class_names) else f"class_{i}", 'confidence': float(probs[i])} for i in top_indices]

    # Grad‑CAM / visual attention overlay
    grad_cam_b64 = get_gradcam(model, tensor, pred_idx, backbone, image_path)

    # SHAP explanation (optional, heavy computation)
    shap_b64 = ''
    if include_shap:
        try:
            train_dir = Path(__file__).resolve().parents[1] / 'train'
            if train_dir.exists():
                train_loader, _, _ = get_data_loaders(str(train_dir), batch_size=4, num_workers=0)
                shap_b64 = get_shap_explanation(model, tensor, train_loader)
        except Exception:
            shap_b64 = ''

    return {
        'label': label,
        'confidence': confidence,
        'top5': top5,
        'grad_cam': grad_cam_b64,
        'shap': shap_b64,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run inference with Grad-CAM & SHAP')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('model_path', type=str, help='Path to .pth checkpoint')
    parser.add_argument('class_names', type=str, help='Comma‑separated list of class names')
    parser.add_argument('--backbone', type=str, default='resnet101', help='Backbone used for the model')
    args = parser.parse_args()
    result = predict(Path(args.image_path), Path(args.model_path), args.class_names.split(','), backbone=args.backbone)
    print(result)
