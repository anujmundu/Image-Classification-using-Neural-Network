# demo/gradio_demo.py
"""Interactive Gradio Web Interface for 11-Architecture Neural Network Benchmark:
1. EfficientNet-B4 (Compound Scaled CNN)
2. MobileNet-V3-Large (Mobile / Edge CNN)
3. ConvNeXt-Tiny (Modernized Pure-CNN)
4. DenseNet-121 (Dense Connectivity CNN)
5. ResNet-101 (Deep Residual CNN)
6. ResNeXt-50 (Cardinality Residual CNN)
7. ShuffleNet-V2 (Channel Shuffle CNN)
8. ViT-Small (Vision Transformer)
9. MLP-Mixer (All-MLP Architecture)
10. CapsNet (Capsule Network)
11. Swin-T (Shifted Window ViT)
Plus:
- Soft-Voting Ensemble (Multi-Model Fusion)
- ONNX Runtime Engine (FP32 & INT8 Quantized)
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from PIL import Image
import gradio as gr

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predict import predict, load_model
from src.ensemble import SoftVotingEnsemble
from src.onnx_engine import ONNXInferenceEngine

MODELS_MAP = {
    "1. EfficientNet-B4 (CNN Champion, 90.71%)": {"type": "pytorch", "backbone": "efficientnet_b4", "ckpt": REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth"},
    "2. MobileNet-V3-Large (Edge CNN, 83.50%)": {"type": "pytorch", "backbone": "mobilenet_v3_large", "ckpt": REPO_ROOT / "models" / "model_mobilenet_v3_large_latest.pth"},
    "3. ConvNeXt-Tiny (Modern CNN, 82.52%)": {"type": "pytorch", "backbone": "convnext_tiny", "ckpt": REPO_ROOT / "models" / "model_convnext_tiny_latest.pth"},
    "4. DenseNet-121 (Dense CNN, 81.28%)": {"type": "pytorch", "backbone": "densenet121", "ckpt": REPO_ROOT / "models" / "model_densenet121_latest.pth"},
    "5. ResNet-101 (Deep ResNet, 81.28%)": {"type": "pytorch", "backbone": "resnet101", "ckpt": REPO_ROOT / "models" / "model_resnet101_latest.pth"},
    "6. ResNeXt-50 (Cardinality CNN, 80.58%)": {"type": "pytorch", "backbone": "resnext50_32x4d", "ckpt": REPO_ROOT / "models" / "model_resnext50_32x4d_latest.pth"},
    "7. ShuffleNet-V2 (Lightweight CNN, 79.47%)": {"type": "pytorch", "backbone": "shufflenet_v2_x1_0", "ckpt": REPO_ROOT / "models" / "model_shufflenet_v2_x1_0_latest.pth"},
    "8. ViT-Small (Vision Transformer, 38.28%)": {"type": "pytorch", "backbone": "vit_small", "ckpt": REPO_ROOT / "models" / "model_vit_small_latest.pth"},
    "9. MLP-Mixer (All-MLP, 27.88%)": {"type": "pytorch", "backbone": "mlp_mixer", "ckpt": REPO_ROOT / "models" / "model_mlp_mixer_latest.pth"},
    "10. CapsNet (Capsule Network, 22.61%)": {"type": "pytorch", "backbone": "capsnet", "ckpt": REPO_ROOT / "models" / "model_capsnet_latest.pth"},
    "11. Swin-T (Shifted Window ViT, 5.96%)": {"type": "pytorch", "backbone": "swin_t", "ckpt": REPO_ROOT / "models" / "model_swin_t_latest.pth"},
    "🤝 Soft-Voting Ensemble (98.92% Top-5)": {"type": "ensemble"},
    "⚡ ONNX Runtime FP32 (18.9 ms / image)": {"type": "onnx", "quantized": False},
    "💾 ONNX Runtime INT8 (17.5 MB Footprint)": {"type": "onnx", "quantized": True},
}

classes_file = REPO_ROOT / "models" / "classes.json"
if classes_file.exists():
    try:
        with open(classes_file, "r") as f:
            CLASS_NAMES = json.load(f)
    except Exception:
        CLASS_NAMES = [f"class_{i}" for i in range(50)]
else:
    CLASS_NAMES = [f"class_{i}" for i in range(50)]

# In-memory caches for fast switching
CACHED_MODELS = {}
CACHED_ENSEMBLE = None
CACHED_ONNX = {}

def get_or_load_model(backbone: str, ckpt_path: Path):
    if backbone not in CACHED_MODELS:
        if ckpt_path.exists():
            CACHED_MODELS[backbone] = load_model(ckpt_path, len(CLASS_NAMES), backbone)
    return CACHED_MODELS.get(backbone)

def get_or_load_ensemble():
    global CACHED_ENSEMBLE
    if CACHED_ENSEMBLE is None:
        configs = {
            "efficientnet_b4": REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth",
            "mobilenet_v3_large": REPO_ROOT / "models" / "model_mobilenet_v3_large_latest.pth",
            "convnext_tiny": REPO_ROOT / "models" / "model_convnext_tiny_latest.pth",
            "resnet101": REPO_ROOT / "models" / "model_resnet101_latest.pth",
        }
        available = {k: v for k, v in configs.items() if v.exists()}
        weights = {"efficientnet_b4": 0.40, "mobilenet_v3_large": 0.25, "convnext_tiny": 0.20, "resnet101": 0.15}
        if available:
            CACHED_ENSEMBLE = SoftVotingEnsemble.from_checkpoints(available, num_classes=len(CLASS_NAMES), weights=weights)
    return CACHED_ENSEMBLE

def get_or_load_onnx(quantized: bool):
    key = "int8" if quantized else "fp32"
    if key not in CACHED_ONNX:
        filename = "model_efficientnet_b4_int8.onnx" if quantized else "model_efficientnet_b4_fp32.onnx"
        path = REPO_ROOT / "models" / filename
        if path.exists():
            CACHED_ONNX[key] = ONNXInferenceEngine(path)
    return CACHED_ONNX.get(key)

def infer(image, model_choice):
    if image is None:
        return None, "Please upload an image.", None, ""
    
    choice_info = MODELS_MAP.get(model_choice, list(MODELS_MAP.values())[0])
    choice_type = choice_info["type"]

    pil_img = Image.fromarray(image.astype("uint8"), "RGB")
    label_conf_dict = {}

    if choice_type == "ensemble":
        ens = get_or_load_ensemble()
        if not ens:
            return None, "Error: Checkpoints for ensemble not found.", None, ""
        start_t = time.perf_counter()
        res = ens.predict_image(pil_img, CLASS_NAMES)
        lat_ms = (time.perf_counter() - start_t) * 1000.0
        for item in res.get("top_rankings", [])[:5]:
            label_conf_dict[item["class"]] = float(item["confidence_pct"] / 100.0)
        top_text = f"Top Prediction (Ensemble): {res['prediction'].upper()} ({res['confidence_pct']}%) | Latency: {lat_ms:.1f} ms"
        return label_conf_dict, top_text, None, f"{lat_ms:.1f} ms"

    elif choice_type == "onnx":
        quantized = choice_info["quantized"]
        engine = get_or_load_onnx(quantized)
        if not engine:
            return None, "Error: ONNX model file not found.", None, ""
        start_t = time.perf_counter()
        res = engine.predict(pil_img, CLASS_NAMES)
        lat_ms = (time.perf_counter() - start_t) * 1000.0
        for item in res.get("top_rankings", [])[:5]:
            label_conf_dict[item["class"]] = float(item["confidence_pct"] / 100.0)
        mode_str = "INT8 Quantized" if quantized else "FP32 Engine"
        top_text = f"Top Prediction (ONNX {mode_str}): {res['prediction'].upper()} ({res['confidence_pct']}%) | Latency: {lat_ms:.1f} ms"
        return label_conf_dict, top_text, None, f"{lat_ms:.1f} ms"

    else:
        backbone = choice_info["backbone"]
        ckpt_path = choice_info["ckpt"]
        if not ckpt_path.exists():
            return None, f"Error: Checkpoint for {model_choice} not found.", None, ""

        tmp_path = Path(tempfile.gettempdir()) / "gradio_upload.png"
        pil_img.save(tmp_path)

        m = get_or_load_model(backbone, ckpt_path)
        start_t = time.perf_counter()
        result = predict(tmp_path, ckpt_path, CLASS_NAMES, backbone=backbone, loaded_model=m)
        lat_ms = (time.perf_counter() - start_t) * 1000.0

        if result.get("top5"):
            for item in result["top5"]:
                label_conf_dict[item["label"]] = float(item["confidence"])
        else:
            label_conf_dict[result["label"]] = float(result["confidence"])

        top_text = f"Top Prediction: {result['label'].upper()} ({result['confidence']*100:.2f}%) | Latency: {lat_ms:.1f} ms"
        grad_cam_img = f"data:image/png;base64,{result['grad_cam']}" if result.get("grad_cam") else None

        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        return label_conf_dict, top_text, grad_cam_img, f"{lat_ms:.1f} ms"

# Build sample image example paths
sample_dir = REPO_ROOT / "sample_images"
example_files = [str(p) for p in sorted(sample_dir.glob("sample_*.jpg"))[:5] if p.exists()]

# Build Gradio interface
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="numpy", label="Upload Input Image"),
        gr.Dropdown(
            choices=list(MODELS_MAP.keys()),
            value="1. EfficientNet-B4 (CNN Champion, 90.71%)",
            label="Neural Network Architecture Paradigm / Engine"
        ),
    ],
    outputs=[
        gr.Label(num_top_classes=5, label="Top-5 Predictions & Probability"),
        gr.Textbox(label="Top Prediction Summary"),
        gr.Image(type="auto", label="Visual Attention / Grad-CAM Heatmap"),
        gr.Textbox(label="Inference Latency"),
    ],
    title="🧠 11-Architecture Neural Network Vision Benchmark",
    description=f"Comparative evaluation across 11 Deep Learning architectures, Soft-Voting Ensemble, and ONNX Runtime on Caltech-50 ({len(CLASS_NAMES)} classes).",
    examples=example_files if example_files else None,
)

if __name__ == "__main__":
    demo.launch()
