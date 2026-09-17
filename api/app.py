# api/app.py
"""FastAPI Production Service for 11-Architecture Neural Network Benchmark:
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
- Multi-Model Soft-Voting Ensemble endpoint
- ONNX Runtime Accelerated Inference endpoint
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predict import predict, load_model
from src.ensemble import SoftVotingEnsemble
from src.onnx_engine import ONNXInferenceEngine

app = FastAPI(
    title="11-Architecture Vision Benchmark API",
    description="Production inference & benchmarking API across 11 Deep Learning architectures, Soft-Voting Ensemble, and ONNX Runtime on Caltech-50.",
    version="2.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of prediction requests', ['endpoint', 'backbone'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of prediction requests', ['endpoint', 'backbone'])

# 11 Official Benchmark Models
CHAMPIONS = {
    "efficientnet_b4": REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth",
    "mobilenet_v3_large": REPO_ROOT / "models" / "model_mobilenet_v3_large_latest.pth",
    "convnext_tiny": REPO_ROOT / "models" / "model_convnext_tiny_latest.pth",
    "densenet121": REPO_ROOT / "models" / "model_densenet121_latest.pth",
    "resnet101": REPO_ROOT / "models" / "model_resnet101_latest.pth",
    "resnext50_32x4d": REPO_ROOT / "models" / "model_resnext50_32x4d_latest.pth",
    "shufflenet_v2_x1_0": REPO_ROOT / "models" / "model_shufflenet_v2_x1_0_latest.pth",
    "vit_small": REPO_ROOT / "models" / "model_vit_small_latest.pth",
    "mlp_mixer": REPO_ROOT / "models" / "model_mlp_mixer_latest.pth",
    "capsnet": REPO_ROOT / "models" / "model_capsnet_latest.pth",
    "swin_t": REPO_ROOT / "models" / "model_swin_t_latest.pth",
}

# Dynamic classes loading
classes_file = REPO_ROOT / "models" / "classes.json"
if classes_file.exists():
    try:
        with open(classes_file, "r") as f:
            CLASS_NAMES = json.load(f)
    except Exception:
        CLASS_NAMES = [f"class_{i}" for i in range(50)]
else:
    CLASS_NAMES = [f"class_{i}" for i in range(50)]

# In-memory model cache
LOADED_MODELS: Dict[str, object] = {}
LOADED_ENSEMBLE: Optional[SoftVotingEnsemble] = None
LOADED_ONNX: Dict[str, ONNXInferenceEngine] = {}

def get_model(backbone: str):
    if backbone not in LOADED_MODELS:
        ckpt_path = CHAMPIONS.get(backbone)
        if ckpt_path and ckpt_path.exists():
            LOADED_MODELS[backbone] = load_model(ckpt_path, len(CLASS_NAMES), backbone)
    return LOADED_MODELS.get(backbone)

def get_ensemble():
    global LOADED_ENSEMBLE
    if LOADED_ENSEMBLE is None:
        configs = {
            "efficientnet_b4": REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth",
            "mobilenet_v3_large": REPO_ROOT / "models" / "model_mobilenet_v3_large_latest.pth",
            "convnext_tiny": REPO_ROOT / "models" / "model_convnext_tiny_latest.pth",
            "resnet101": REPO_ROOT / "models" / "model_resnet101_latest.pth",
        }
        available = {k: v for k, v in configs.items() if v.exists()}
        weights = {"efficientnet_b4": 0.40, "mobilenet_v3_large": 0.25, "convnext_tiny": 0.20, "resnet101": 0.15}
        if available:
            LOADED_ENSEMBLE = SoftVotingEnsemble.from_checkpoints(available, num_classes=len(CLASS_NAMES), weights=weights)
    return LOADED_ENSEMBLE

def get_onnx_engine(quantized: bool = False):
    key = "int8" if quantized else "fp32"
    if key not in LOADED_ONNX:
        filename = "model_efficientnet_b4_int8.onnx" if quantized else "model_efficientnet_b4_fp32.onnx"
        path = REPO_ROOT / "models" / filename
        if path.exists():
            LOADED_ONNX[key] = ONNXInferenceEngine(path)
    return LOADED_ONNX.get(key)

class TopPrediction(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    backbone: str
    label: str
    confidence: float
    latency_ms: float
    top5: List[TopPrediction]
    grad_cam: Optional[str] = ""

@app.get("/")
def root():
    return {
        "service": "11-Architecture Neural Network Vision Benchmark API",
        "dataset": "Caltech-50 (50 curated classes)",
        "available_architectures": list(CHAMPIONS.keys()),
        "ensemble_available": True,
        "onnx_available": True,
        "docs": "/docs",
        "leaderboard": "/benchmark/leaderboard",
    }

@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/benchmark/leaderboard")
def get_leaderboard():
    md_path = REPO_ROOT / "Result" / "benchmark_leaderboard.md"
    json_path = REPO_ROOT / "Result" / "benchmark_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            return {"markdown": f.read()}
    raise HTTPException(status_code=404, detail="Benchmark results not found.")

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    backbone: str = Query("efficientnet_b4", enum=list(CHAMPIONS.keys()), description="Select champion architecture")
):
    if backbone not in CHAMPIONS:
        raise HTTPException(status_code=400, detail=f"Invalid backbone. Choose from {list(CHAMPIONS.keys())}")
    
    ckpt_path = CHAMPIONS[backbone]
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Model checkpoint for {backbone} not found on disk.")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file format. Upload JPEG or PNG.")

    REQUEST_COUNT.labels(endpoint="/predict", backbone=backbone).inc()
    start = time.perf_counter()

    temp_path = Path(tempfile.gettempdir()) / file.filename
    contents = await file.read()
    with open(temp_path, "wb") as out:
        out.write(contents)

    try:
        model = get_model(backbone)
        res = predict(temp_path, ckpt_path, CLASS_NAMES, backbone=backbone, loaded_model=model)
        latency_ms = (time.perf_counter() - start) * 1000.0
        REQUEST_LATENCY.labels(endpoint="/predict", backbone=backbone).observe(latency_ms / 1000.0)

        return PredictionResponse(
            backbone=backbone,
            label=res["label"],
            confidence=res["confidence"],
            latency_ms=round(latency_ms, 2),
            top5=[TopPrediction(label=t["label"], confidence=t["confidence"]) for t in res.get("top5", [])],
            grad_cam=res.get("grad_cam", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(file: UploadFile = File(...)):
    """Soft-Voting Multi-Model Ensemble combining top vision models (98.92% Top-5)."""
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file format. Upload JPEG or PNG.")

    ens = get_ensemble()
    if not ens:
        raise HTTPException(status_code=503, detail="Ensemble models not available.")

    REQUEST_COUNT.labels(endpoint="/predict/ensemble", backbone="ensemble").inc()
    start = time.perf_counter()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        res = ens.predict_image(image, CLASS_NAMES)
        latency_ms = (time.perf_counter() - start) * 1000.0
        REQUEST_LATENCY.labels(endpoint="/predict/ensemble", backbone="ensemble").observe(latency_ms / 1000.0)

        top5 = [
            TopPrediction(label=r["class"], confidence=round(r["confidence_pct"] / 100.0, 4))
            for r in res.get("top_rankings", [])[:5]
        ]

        return PredictionResponse(
            backbone="soft_voting_ensemble",
            label=res["prediction"],
            confidence=round(res["confidence_pct"] / 100.0, 4),
            latency_ms=round(latency_ms, 2),
            top5=top5,
            grad_cam=""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/onnx", response_model=PredictionResponse)
async def predict_onnx(
    file: UploadFile = File(...),
    quantized: bool = Query(False, description="Use INT8 quantized ONNX engine (17.5 MB) or FP32 engine (18.9 ms)")
):
    """High-Speed Hardware Accelerated ONNX Runtime Prediction (3.01x Speedup)."""
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file format. Upload JPEG or PNG.")

    engine = get_onnx_engine(quantized=quantized)
    if not engine:
        raise HTTPException(status_code=503, detail="ONNX runtime engine not available.")

    engine_tag = "onnx_int8" if quantized else "onnx_fp32"
    REQUEST_COUNT.labels(endpoint="/predict/onnx", backbone=engine_tag).inc()
    start = time.perf_counter()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        res = engine.predict(image, CLASS_NAMES)
        latency_ms = (time.perf_counter() - start) * 1000.0
        REQUEST_LATENCY.labels(endpoint="/predict/onnx", backbone=engine_tag).observe(latency_ms / 1000.0)

        top5 = [
            TopPrediction(label=r["class"], confidence=round(r["confidence_pct"] / 100.0, 4))
            for r in res.get("top_rankings", [])[:5]
        ]

        return PredictionResponse(
            backbone=engine_tag,
            label=res["prediction"],
            confidence=round(res["confidence_pct"] / 100.0, 4),
            latency_ms=round(latency_ms, 2),
            top5=top5,
            grad_cam=""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
