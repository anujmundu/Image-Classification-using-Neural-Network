# scripts/generate_benchmark_leaderboard.py
"""Benchmark Leaderboard Generator for 4 Deep Learning Paradigms:
1. Convolutional Neural Network (CNN): EfficientNet-B4
2. Vision Transformer (ViT): ViT-Small
3. All-MLP Architecture: MLP-Mixer
4. Specialized Architecture: Capsule Network (CapsNet)
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.model import build_model

CHAMPIONS = [
    {
        "category": "CNN (Compound Scaling)",
        "model_name": "EfficientNet-B4",
        "backbone": "efficientnet_b4",
        "description": "Deep hierarchical spatial convolutions with compound scaling",
    },
    {
        "category": "CNN (Residual)",
        "model_name": "ResNet-101",
        "backbone": "resnet101",
        "description": "Deep 101-layer residual identity shortcut learning",
    },
    {
        "category": "Vision Transformer (ViT)",
        "model_name": "ViT-Small",
        "backbone": "vit_small",
        "description": "Non-convolutional patch projection with multi-head self-attention",
    },
    {
        "category": "All-MLP Architecture",
        "model_name": "MLP-Mixer",
        "backbone": "mlp_mixer",
        "description": "Pure MLPs with alternating spatial token and channel mixing",
    },
    {
        "category": "Specialized / Experimental",
        "model_name": "Capsule Network (CapsNet)",
        "backbone": "capsnet",
        "description": "Spatial vector capsules with pose invariance aggregation",
    },
]

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def measure_inference_speed(model: torch.nn.Module, device: torch.device, img_size: int = 224, warmup: int = 10, runs: int = 40):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    latency_ms = (total_time / runs) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return latency_ms, fps

def generate_comparison_plots(results: list, output_path: Path):
    """Generate high-resolution comparative bar plots for Accuracy, F1, Latency, and Parameters."""
    models = [r["model_name"] for r in results]
    accs = [r["val_accuracy_raw"] * 100 for r in results]
    f1s = [r["val_f1_raw"] * 100 for r in results]
    params = [r["params_m_raw"] for r in results]
    latencies = [r["latency_ms_raw"] for r in results]

    x = np.arange(len(models))
    colors = ["#2563eb", "#3b82f6", "#7c3aed", "#059669", "#d97706"][:len(models)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
    fig.suptitle("5-Model Multi-Class Deep Learning Benchmark (Curated 50 Categories)", fontsize=16, fontweight="bold", y=0.98)

    # 1. Validation Accuracy
    bars1 = ax1.bar(x, accs, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_title("Top-1 Validation Accuracy (%) - Higher is Better", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax1.set_ylim(0, max(accs) * 1.18)

    # 2. Validation F1 Score
    bars2 = ax2.bar(x, f1s, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_title("Weighted F1 Score (%) - Higher is Better", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel("Weighted F1 (%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax2.set_ylim(0, max(f1s) * 1.18)

    # 3. Parameter Size (M)
    bars3 = ax3.bar(x, params, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax3.set_title("Parameter Count (Millions)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax3.set_ylabel("Parameters (M)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax3.set_ylim(0, max(params) * 1.18)

    # 4. Inference Latency (ms)
    bars4 = ax4.bar(x, latencies, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax4.set_title("Inference Latency per Image (ms) - Lower is Better", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax4.set_ylabel("Latency (ms)")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}ms", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax4.set_ylim(0, max(latencies) * 1.18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOTS] Comparison charts saved to {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"4-CATEGORY BENCHMARK LEADERBOARD EVALUATION")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    models_dir = REPO_ROOT / "models"
    result_dir = REPO_ROOT / "Result"
    result_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for champ in CHAMPIONS:
        backbone = champ["backbone"]
        ckpt_path = models_dir / f"model_{backbone}_latest.pth"
        meta_path = models_dir / f"metadata_{backbone}.json"

        if not ckpt_path.exists():
            print(f"⚠️ Warning: Checkpoint {ckpt_path.name} not found. Skipping.")
            continue

        meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)

        num_classes = meta.get("num_classes", 257)
        print(f"Loading {champ['model_name']} ({backbone})...", flush=True)

        model = build_model(num_classes=num_classes, backbone=backbone)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        total_params = count_parameters(model)
        file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)

        latency_ms, fps = measure_inference_speed(model, device)

        val_acc = meta.get("val_accuracy", 0.0)
        val_f1 = meta.get("val_f1", 0.0)
        train_loss = meta.get("train_loss", 0.0)
        epochs = meta.get("epochs", 10)

        record = {
            "category": champ["category"],
            "model_name": champ["model_name"],
            "backbone": backbone,
            "description": champ["description"],
            "params_m": f"{total_params / 1e6:.2f}M",
            "params_m_raw": total_params / 1e6,
            "val_accuracy": f"{val_acc * 100:.2f}%",
            "val_accuracy_raw": val_acc,
            "val_f1": f"{val_f1:.4f}",
            "val_f1_raw": val_f1,
            "train_loss": f"{train_loss:.4f}",
            "train_loss_raw": train_loss,
            "latency_ms": f"{latency_ms:.2f} ms",
            "latency_ms_raw": latency_ms,
            "fps": f"{fps:.1f} FPS",
            "fps_raw": fps,
            "checkpoint_size": f"{file_size_mb:.1f} MB",
            "epochs": epochs,
            "classes": num_classes,
        }
        results.append(record)
        print(f"  -> Params: {record['params_m']} | Acc: {record['val_accuracy']} | F1: {record['val_f1']} | Latency: {record['latency_ms']} ({record['fps']})")

    # Sort results by validation accuracy descending
    results.sort(key=lambda r: r["val_accuracy_raw"], reverse=True)

    # 1. Save JSON
    json_path = result_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved results JSON to {json_path}")

    # 2. Generate Markdown Leaderboard
    md_lines = [
        "# 🏆 4-Category Multi-Class Deep Learning Benchmark Leaderboard\n",
        f"**Dataset:** Caltech-256 (257 Object Categories)  ",
        f"**Hardware Platform:** {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} (Tensor Core AMP Active)  ",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n",
        "| Rank | Category | Architecture | Backbone Flag | Parameters | Val Accuracy (Top-1) | Val F1 Score | Train Loss | Latency (ms) | Throughput (FPS) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for idx, r in enumerate(results, start=1):
        md_lines.append(
            f"| **#{idx}** | **{r['category']}** | **{r['model_name']}** | `{r['backbone']}` | {r['params_m']} | **{r['val_accuracy']}** | {r['val_f1']} | {r['train_loss']} | {r['latency_ms']} | {r['fps']} |"
        )

    md_lines.append("\n## Architectural Breakdown & Insights\n")
    for r in results:
        md_lines.append(f"- **{r['model_name']} ({r['category']}):** {r['description']}. Achieved {r['val_accuracy']} accuracy with {r['params_m']} parameters and {r['latency_ms']} latency.")

    md_path = result_dir / "benchmark_leaderboard.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Saved Markdown Leaderboard to {md_path}")

    # 3. Generate Visual Comparison Plots
    plot_path = result_dir / "benchmark_comparison_charts.png"
    generate_comparison_plots(results, plot_path)

    print("\n" + "=" * 80)
    try:
        print(md_path.read_text(encoding="utf-8"))
    except UnicodeEncodeError:
        print(md_path.read_text(encoding="utf-8").encode("ascii", "replace").decode("ascii"))
    print("=" * 80)

if __name__ == "__main__":
    main()
