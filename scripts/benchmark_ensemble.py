# scripts/benchmark_ensemble.py
"""Benchmarks the Multi-Model Soft Voting Ensemble against individual standalone models on the 50-class test split."""

import sys
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import get_data_loaders
from src.model import build_model
from src.ensemble import SoftVotingEnsemble

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_classes():
    classes_path = REPO_ROOT / 'models' / 'classes.json'
    if not classes_path.exists():
        classes_path = REPO_ROOT / 'data' / 'curated_benchmark_50' / 'classes.json'
    with open(classes_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_predictions(probs: np.ndarray, y_true: np.ndarray):
    """Compute Top-1 Accuracy, Top-5 Accuracy, and Macro F1."""
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred) * 100.0
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    top5_correct = 0
    for i in range(len(y_true)):
        if y_true[i] in np.argsort(probs[i])[-5:]:
            top5_correct += 1
    top5_acc = (top5_correct / len(y_true)) * 100.0
    
    return acc, top5_acc, f1 * 100.0

def main():
    print("=" * 80)
    print("BENCHMARKING MULTI-MODEL SOFT VOTING ENSEMBLE")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    classes = load_classes()
    num_classes = len(classes)
    
    # Load test split
    _, _, test_loader = get_data_loaders(
        data_dir='data/curated_benchmark_50',
        batch_size=32,
        img_size=224,
        num_workers=0
    )
    
    models_config = [
        ("EfficientNet-B4", "efficientnet_b4", 0.55),
        ("ResNet-101", "resnet101", 0.30),
        ("ViT-Small", "vit_small", 0.15),
    ]
    
    loaded_models = []
    model_names = []
    weights = []
    individual_probs = {}
    
    for name, backbone, weight in models_config:
        ckpt_path = REPO_ROOT / "models" / f"model_{backbone}_latest.pth"
        if not ckpt_path.exists():
            print(f"⚠️ Checkpoint for {name} ({ckpt_path.name}) not found. Skipping.")
            continue
        print(f"Loading {name} ({backbone})...", flush=True)
        m = build_model(num_classes=num_classes, backbone=backbone)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        loaded_models.append(m)
        model_names.append(name)
        weights.append(weight)
        
    if not loaded_models:
        print("❌ No models could be loaded. Exiting.")
        return
        
    ensemble = SoftVotingEnsemble(loaded_models, weights=weights, model_names=model_names)
    ensemble.to(DEVICE)
    ensemble.eval()
    
    # Run test evaluation
    all_targets = []
    all_ensemble_probs = []
    
    for name in model_names:
        individual_probs[name] = []
        
    print("\n>> Evaluating models on test split...", flush=True)
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            all_targets.extend(targets.numpy())
            
            # Evaluate individual models
            for m, name in zip(loaded_models, model_names):
                logits = m(inputs)
                p = torch.softmax(logits, dim=1)
                individual_probs[name].append(p.cpu().numpy())
                
            # Evaluate ensemble
            ens_p = ensemble(inputs)
            all_ensemble_probs.append(ens_p.cpu().numpy())
            
    y_true = np.array(all_targets)
    ensemble_p = np.concatenate(all_ensemble_probs, axis=0)
    
    for name in model_names:
        individual_probs[name] = np.concatenate(individual_probs[name], axis=0)
        
    # Compute Metrics
    summary = []
    for name in model_names:
        acc, top5, f1 = evaluate_predictions(individual_probs[name], y_true)
        summary.append({
            "model": name,
            "type": "Standalone",
            "top1_acc": acc,
            "top5_acc": top5,
            "macro_f1": f1
        })
        
    ens_acc, ens_top5, ens_f1 = evaluate_predictions(ensemble_p, y_true)
    summary.append({
        "model": "Soft-Voting Ensemble (EffNet + ResNet + ViT)",
        "type": "Ensemble",
        "top1_acc": ens_acc,
        "top5_acc": ens_top5,
        "macro_f1": ens_f1
    })
    
    print("\n" + "=" * 90)
    print(f"{'Model Architecture':<48} | {'Top-1 Acc':<11} | {'Top-5 Acc':<11} | {'Macro F1':<10}")
    print("=" * 90)
    for s in summary:
        print(f"{s['model']:<48} | {s['top1_acc']:>8.2f}% | {s['top5_acc']:>8.2f}% | {s['macro_f1']:>8.2f}%")
    print("=" * 90)
    
    # Save JSON results
    out_dir = REPO_ROOT / "Result"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ensemble_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Results saved to {json_path}")
    
    # Generate Visual Plot
    plot_path = out_dir / "Ensemble_vs_Individual_Models_Comparison.png"
    
    names = [s["model"] if len(s["model"]) < 25 else "Soft Voting Ensemble" for s in summary]
    top1s = [s["top1_acc"] for s in summary]
    top5s = [s["top5_acc"] for s in summary]
    
    x = np.arange(len(names))
    width = 0.35
    colors_top1 = ['#3b82f6', '#60a5fa', '#a78bfa', '#059669']
    colors_top5 = ['#1d4ed8', '#2563eb', '#7c3aed', '#047857']
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bars1 = ax.bar(x - width/2, top1s, width, label='Top-1 Accuracy (%)', color=colors_top1, edgecolor='black')
    bars2 = ax.bar(x + width/2, top5s, width, label='Top-5 Accuracy (%)', color=colors_top5, edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Soft Voting Ensemble vs Individual Standalone Architectures (50 Classes)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    
    for bar in bars1:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 1.2, f'{y:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 1.2, f'{y:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
