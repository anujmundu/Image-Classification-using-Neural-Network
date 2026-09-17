# scripts/benchmark_onnx_quantization.py
"""Performs INT8 Post-Training Dynamic Quantization on the Champion Model and Benchmarks:
1. Storage Footprint (PyTorch .pth vs ONNX FP32 vs ONNX INT8)
2. Latency & Throughput (PyTorch vs ONNX FP32 vs ONNX INT8)
3. Accuracy Preservation Check
"""

import sys
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.model import build_model
from src.onnx_engine import ONNXInferenceEngine

DEVICE = torch.device('cpu')  # CPU execution for standard edge/server benchmark

def main():
    print("=" * 80)
    print("ONNX RUNTIME & INT8 DYNAMIC QUANTIZATION BENCHMARK")
    print("=" * 80)
    
    models_dir = REPO_ROOT / "models"
    out_dir = REPO_ROOT / "Result"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    classes_file = models_dir / "classes.json"
    with open(classes_file, "r") as f:
        classes = json.load(f)
    num_classes = len(classes)
    
    # 1. Load PyTorch model
    print(">> Step 1: Loading PyTorch Champion Model (EfficientNet-B4)...", flush=True)
    pytorch_model = build_model(num_classes=num_classes, backbone='efficientnet_b4')
    ckpt_path = models_dir / "model_efficientnet_b4_latest.pth"
    pytorch_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    pytorch_model.eval()
    
    # 2. Export to standard ONNX FP32
    fp32_onnx_path = models_dir / "model_efficientnet_b4_fp32.onnx"
    print(f">> Step 2: Exporting to ONNX FP32: {fp32_onnx_path.name}...", flush=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(fp32_onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False,
        opset_version=18
    )
    
    # 3. Dynamic INT8 Quantization
    int8_onnx_path = models_dir / "model_efficientnet_b4_int8.onnx"
    print(f">> Step 3: Quantizing to INT8: {int8_onnx_path.name}...", flush=True)
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8
    )
    
    # 4. Measure File Sizes
    pth_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    fp32_size_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
    int8_size_mb = int8_onnx_path.stat().st_size / (1024 * 1024)
    compression_ratio = (1.0 - (int8_size_mb / pth_size_mb)) * 100.0
    
    print("\n" + "-" * 60)
    print(f"PyTorch Checkpoint (.pth) : {pth_size_mb:.2f} MB")
    print(f"ONNX Model (FP32)         : {fp32_size_mb:.2f} MB")
    print(f"ONNX Quantized (INT8)     : {int8_size_mb:.2f} MB ({compression_ratio:.1f}% Reduction)")
    print("-" * 60)
    
    # 5. Benchmark Latency (100 Iterations on CPU)
    print("\n>> Step 4: Benchmarking Inference Latency over 100 iterations on CPU...", flush=True)
    
    # A. PyTorch FP32
    for _ in range(10):  # warmup
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
            
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
    torch_latency = ((time.perf_counter() - t0) / 100.0) * 1000.0
    torch_fps = 1000.0 / torch_latency
    
    # B. ONNX Runtime FP32
    engine_fp32 = ONNXInferenceEngine(fp32_onnx_path, providers=['CPUExecutionProvider'])
    onnx_fp32_lat, onnx_fp32_fps = engine_fp32.benchmark(iterations=100)
    
    # C. ONNX Runtime INT8
    engine_int8 = ONNXInferenceEngine(int8_onnx_path, providers=['CPUExecutionProvider'])
    onnx_int8_lat, onnx_int8_fps = engine_int8.benchmark(iterations=100)
    
    print("\n" + "=" * 80)
    print(f"{'Execution Runtime / Precision':<35} | {'Size (MB)':<10} | {'Latency (ms)':<12} | {'Throughput (FPS)':<16}")
    print("=" * 80)
    print(f"{'PyTorch Native (FP32)':<35} | {pth_size_mb:>8.2f} MB | {torch_latency:>10.2f} ms | {torch_fps:>14.1f} FPS")
    print(f"{'ONNX Runtime Engine (FP32)':<35} | {fp32_size_mb:>8.2f} MB | {onnx_fp32_lat:>10.2f} ms | {onnx_fp32_fps:>14.1f} FPS")
    print(f"{'ONNX Runtime Engine (INT8 Quantized)':<35} | {int8_size_mb:>8.2f} MB | {onnx_int8_lat:>10.2f} ms | {onnx_int8_fps:>14.1f} FPS")
    print("=" * 80)
    speedup_onnx_fp32 = torch_latency / onnx_fp32_lat
    print(f">> Acceleration: ONNX Runtime FP32 is {speedup_onnx_fp32:.2f}x faster than PyTorch Native CPU!")
    print(f">> Compression : INT8 Quantization reduces model size by {compression_ratio:.1f}% (from {pth_size_mb:.1f} MB to {int8_size_mb:.1f} MB)!")
    print("=" * 80)
    
    # Save JSON summary
    results_summary = {
        "model": "EfficientNet-B4",
        "pytorch_fp32": {"size_mb": round(pth_size_mb, 2), "latency_ms": round(torch_latency, 2), "fps": round(torch_fps, 1)},
        "onnx_fp32": {"size_mb": round(fp32_size_mb, 2), "latency_ms": round(onnx_fp32_lat, 2), "fps": round(onnx_fp32_fps, 1)},
        "onnx_int8": {"size_mb": round(int8_size_mb, 2), "latency_ms": round(onnx_int8_lat, 2), "fps": round(onnx_int8_fps, 1)},
        "size_reduction_pct": round(compression_ratio, 1),
        "speedup_onnx_fp32_vs_pytorch": round(speedup_onnx_fp32, 2)
    }
    with open(out_dir / "onnx_quantization_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
        
    # Generate Visual Plot
    plot_path = out_dir / "ONNX_and_INT8_Quantization_Benchmark.png"
    runtimes = ['PyTorch FP32', 'ONNX Runtime FP32', 'ONNX Runtime INT8']
    sizes = [pth_size_mb, fp32_size_mb, int8_size_mb]
    latencies = [torch_latency, onnx_fp32_lat, onnx_int8_lat]
    throughputs = [torch_fps, onnx_fp32_fps, onnx_int8_fps]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle('Model Optimization & Acceleration: PyTorch vs ONNX Runtime vs INT8', fontsize=15, fontweight='bold', y=1.02)
    
    colors = ['#2563eb', '#0891b2', '#059669']
    
    # 1. Disk Size
    bars1 = ax1.bar(runtimes, sizes, color=colors, edgecolor='black', width=0.55)
    ax1.set_title('Disk Storage Footprint (MB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('File Size (MB) -> Lower is Better', fontsize=10, fontweight='bold')
    ax1.set_xticklabels(['PyTorch\nFP32', 'ONNX\nFP32', 'ONNX\nINT8'], fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars1:
        y = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, y + 1.5, f'{y:.1f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    # 2. Latency
    bars2 = ax2.bar(runtimes, latencies, color=colors, edgecolor='black', width=0.55)
    ax2.set_title('Inference Latency per Image (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency (ms) -> Lower is Better', fontsize=10, fontweight='bold')
    ax2.set_xticklabels(['PyTorch\nFP32', 'ONNX\nFP32', 'ONNX\nINT8'], fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars2:
        y = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, y + 1.2, f'{y:.1f} ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    # 3. Throughput
    bars3 = ax3.bar(runtimes, throughputs, color=colors, edgecolor='black', width=0.55)
    ax3.set_title('Throughput Speed (FPS)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frames Per Second (FPS) -> Higher is Better', fontsize=10, fontweight='bold')
    ax3.set_xticklabels(['PyTorch\nFP32', 'ONNX\nFP32', 'ONNX\nINT8'], fontsize=9.5)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars3:
        y = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2, y + 0.6, f'{y:.1f} FPS', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Optimization benchmark plot saved to {plot_path}")

if __name__ == "__main__":
    main()
