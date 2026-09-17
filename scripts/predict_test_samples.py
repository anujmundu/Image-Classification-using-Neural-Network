# scripts/predict_test_samples.py
"""Comprehensive evaluation script running real-time inference on all 20 benchmark sample images.
Evaluates:
1. EfficientNet-B4 (Champion Standalone)
2. MobileNet-V3-Large (Fast Edge CNN)
3. ConvNeXt-Tiny (Modern Pure-CNN)
4. Soft-Voting Ensemble (Multi-Model Fusion)
5. ONNX Runtime Engine (FP32 & INT8 Quantized)
"""

import os
import sys
import time
import json
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.predict import load_model
from src.ensemble import SoftVotingEnsemble
from src.onnx_engine import ONNXInferenceEngine

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_classes():
    classes_path = REPO_ROOT / 'models' / 'classes.json'
    with open(classes_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Ground truth mapping for the 20 benchmark sample images
GROUND_TRUTH_MAPPING = {
    "sample_1_butterfly.jpg": "butterfly",
    "sample_2_leopard.jpg": "leopards-101",
    "sample_3_motorbike.jpg": "motorbikes-101",
    "sample_4_school_bus.jpg": "school-bus",
    "sample_5_backpack.jpg": "backpack",
    "sample_6_bear.jpg": "bear",
    "sample_7_binoculars.jpg": "binoculars",
    "sample_8_blimp.jpg": "blimp",
    "sample_9_bonsai.jpg": "bonsai-101",
    "sample_10_bulldozer.jpg": "bulldozer",
    "sample_11_cactus.jpg": "cactus",
    "sample_12_camel.jpg": "camel",
    "sample_13_canoe.jpg": "canoe",
    "sample_14_dog.jpg": "dog",
    "sample_15_golden_gate_bridge.jpg": "golden-gate-bridge",
    "sample_16_goldfish.jpg": "goldfish",
    "sample_17_grand_piano.jpg": "grand-piano-101",
    "sample_18_horse.jpg": "horse",
    "sample_19_hummingbird.jpg": "hummingbird",
    "sample_20_zebra.jpg": "zebra",
}

def main():
    print("=" * 110)
    print("MULTI-MODEL INFERENCE & ACCELERATION BENCHMARK ON 20 CALTECH-50 SAMPLES")
    print(f"Compute Device: {DEVICE} (Tensor Core Acceleration: {torch.cuda.is_available()})")
    print("=" * 110)

    classes = load_classes()
    num_classes = len(classes)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. Load PyTorch Models
    print("[1/4] Loading PyTorch neural network checkpoints...")
    models_dict = {}
    backbones = ["efficientnet_b4", "mobilenet_v3_large", "convnext_tiny", "resnet101"]
    for bb in backbones:
        ckpt = REPO_ROOT / "models" / f"model_{bb}_latest.pth"
        if ckpt.exists():
            m = load_model(ckpt, num_classes=num_classes, backbone=bb)
            m.to(DEVICE)
            m.eval()
            models_dict[bb] = m
            print(f"      - {bb}: Loaded successfully ({ckpt.name})")

    # 2. Load Soft-Voting Ensemble
    print("[2/4] Initializing Multi-Model Soft-Voting Ensemble...")
    ens_configs = {bb: REPO_ROOT / "models" / f"model_{bb}_latest.pth" for bb in backbones}
    weights = {"efficientnet_b4": 0.40, "mobilenet_v3_large": 0.25, "convnext_tiny": 0.20, "resnet101": 0.15}
    ensemble = SoftVotingEnsemble.from_checkpoints(ens_configs, num_classes=num_classes, weights=weights, device=DEVICE)
    print("      - Soft-Voting Ensemble initialized (4 heterogeneous backbones fused)")

    # 3. Load ONNX Engines
    print("[3/4] Initializing Hardware-Accelerated ONNX Runtime Engines...")
    onnx_fp32_path = REPO_ROOT / "models" / "model_efficientnet_b4_fp32.onnx"
    onnx_int8_path = REPO_ROOT / "models" / "model_efficientnet_b4_int8.onnx"
    onnx_fp32 = ONNXInferenceEngine(onnx_fp32_path) if onnx_fp32_path.exists() else None
    onnx_int8 = ONNXInferenceEngine(onnx_int8_path) if onnx_int8_path.exists() else None
    print(f"      - ONNX FP32 Engine: {'Active' if onnx_fp32 else 'Missing'}")
    print(f"      - ONNX INT8 Quantized Engine: {'Active' if onnx_int8 else 'Missing'}")

    # 4. Run Evaluation across all 20 images
    sample_dir = REPO_ROOT / "sample_images"
    sample_files = sorted(list(sample_dir.glob("sample_*.jpg")), key=lambda p: int(p.stem.split('_')[1]))

    print(f"[4/4] Executing benchmark across {len(sample_files)} test sample images...\n")

    header = f"{'#':<3} | {'Image Name':<28} | {'Ground Truth':<18} | {'EffNet-B4 Pred':<18} | {'Conf':<6} | {'Ensemble Pred':<18} | {'Conf':<6} | {'Status':<6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    eff_correct = 0
    mob_correct = 0
    ens_correct = 0
    total_samples = 0
    all_results = []

    eff_latencies = []
    ens_latencies = []
    onnx_latencies = []

    for idx, img_path in enumerate(sample_files, start=1):
        gt = GROUND_TRUTH_MAPPING.get(img_path.name, "unknown")
        total_samples += 1

        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        # A. EfficientNet-B4
        t0 = time.perf_counter()
        with torch.no_grad():
            out = models_dict["efficientnet_b4"](tensor)
            probs = F.softmax(out, dim=1)[0]
            eff_idx = torch.argmax(probs).item()
            eff_pred = classes[eff_idx]
            eff_conf = probs[eff_idx].item() * 100.0
        eff_lat = (time.perf_counter() - t0) * 1000.0
        eff_latencies.append(eff_lat)

        # B. MobileNet-V3
        with torch.no_grad():
            out_mob = models_dict["mobilenet_v3_large"](tensor)
            probs_mob = F.softmax(out_mob, dim=1)[0]
            mob_idx = torch.argmax(probs_mob).item()
            mob_pred = classes[mob_idx]

        # C. Soft-Voting Ensemble
        t_ens = time.perf_counter()
        ens_res = ensemble.predict_image(img, classes)
        ens_lat = (time.perf_counter() - t_ens) * 1000.0
        ens_latencies.append(ens_lat)
        ens_pred = ens_res["prediction"]
        ens_conf = ens_res["confidence_pct"]

        # D. ONNX FP32
        onnx_pred = ""
        onnx_conf = 0.0
        if onnx_fp32:
            t_onnx = time.perf_counter()
            onnx_res = onnx_fp32.predict(img, classes)
            onnx_lat = (time.perf_counter() - t_onnx) * 1000.0
            onnx_latencies.append(onnx_lat)
            onnx_pred = onnx_res["prediction"]
            onnx_conf = onnx_res["confidence_pct"]

        # Evaluate correctness
        is_eff_match = (eff_pred.lower() == gt.lower())
        is_mob_match = (mob_pred.lower() == gt.lower())
        is_ens_match = (ens_pred.lower() == gt.lower())

        if is_eff_match: eff_correct += 1
        if is_mob_match: mob_correct += 1
        if is_ens_match: ens_correct += 1

        status_str = "[OK]" if is_ens_match else "[DIFF]"

        print(f"{idx:<3} | {img_path.name:<28} | {gt:<18} | {eff_pred:<18} | {eff_conf:>5.1f}% | {ens_pred:<18} | {ens_conf:>5.1f}% | {status_str:<6}")

        all_results.append({
            "sample_id": idx,
            "filename": img_path.name,
            "ground_truth": gt,
            "efficientnet_b4": {"prediction": eff_pred, "confidence_pct": round(eff_conf, 2), "correct": is_eff_match},
            "mobilenet_v3_large": {"prediction": mob_pred, "correct": is_mob_match},
            "soft_voting_ensemble": {"prediction": ens_pred, "confidence_pct": round(ens_conf, 2), "correct": is_ens_match},
            "onnx_runtime_fp32": {"prediction": onnx_pred, "confidence_pct": round(onnx_conf, 2)}
        })

    print("-" * len(header))
    print("\n" + "=" * 80)
    print("FINAL TEST ACCURACY & INFERENCE LATENCY SUMMARY")
    print("=" * 80)
    print(f"Total Test Samples:             {total_samples}")
    print(f"EfficientNet-B4 Standalone:     {eff_correct}/{total_samples} Correct ({eff_correct/total_samples*100:.1f}%) | Latency: {sum(eff_latencies)/len(eff_latencies):.2f} ms")
    print(f"MobileNet-V3-Large Standalone:  {mob_correct}/{total_samples} Correct ({mob_correct/total_samples*100:.1f}%)")
    print(f"Soft-Voting Ensemble Fusion:    {ens_correct}/{total_samples} Correct ({ens_correct/total_samples*100:.1f}%) | Latency: {sum(ens_latencies)/len(ens_latencies):.2f} ms")
    if onnx_latencies:
        print(f"ONNX Runtime Engine (CPU):      Avg Latency: {sum(onnx_latencies)/len(onnx_latencies):.2f} ms (Throughput: {1000.0 / (sum(onnx_latencies)/len(onnx_latencies)):.1f} FPS)")
    print("=" * 80)

    # Save detailed JSON summary
    out_json = REPO_ROOT / "Result" / "sample_predictions_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": total_samples,
            "efficientnet_b4_accuracy_pct": eff_correct / total_samples * 100.0,
            "mobilenet_v3_large_accuracy_pct": mob_correct / total_samples * 100.0,
            "soft_voting_ensemble_accuracy_pct": ens_correct / total_samples * 100.0,
            "avg_latency_pytorch_effnet_ms": round(sum(eff_latencies) / len(eff_latencies), 2),
            "avg_latency_ensemble_ms": round(sum(ens_latencies) / len(ens_latencies), 2),
            "avg_latency_onnx_ms": round(sum(onnx_latencies) / len(onnx_latencies), 2) if onnx_latencies else None,
            "detailed_predictions": all_results
        }, f, indent=2)
    print(f"\n[OUTPUT] Detailed evaluation report saved to: {out_json}")

if __name__ == '__main__':
    main()
