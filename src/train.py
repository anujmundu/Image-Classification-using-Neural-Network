import argparse
import os
import sys
import time
import json
from pathlib import Path

# Ensure repository root is in sys.path when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import get_data_loaders
from src.model import build_model
import mlflow
import optuna
from tqdm import tqdm

def objective(trial, train_loader, val_loader, num_classes, device):
    # Hyperparameters to optimise
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    backbone = trial.suggest_categorical('backbone', ['resnet101', 'efficientnet_b4'])

    model = build_model(num_classes=num_classes, backbone=backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Simple training loop (1 epoch for trial)
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx > 100:  # limit steps for speed
            break

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    acc = accuracy_score(y_true, y_pred)
    trial.set_user_attr('val_accuracy', acc)
    return 1.0 - acc  # minimise error

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"[GPU] Using GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)", flush=True)
    else:
        print("[WARNING] PyTorch is running on CPU. For fast training on your NVIDIA RTX GPU, install the CUDA build of PyTorch.", flush=True)
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
    )
    if hasattr(train_loader.dataset, 'classes'):
        classes = train_loader.dataset.classes
    elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'classes'):
        classes = train_loader.dataset.dataset.classes
    else:
        classes = [f"class_{i}" for i in range(10)]
    num_classes = len(classes)
    print(f"[DATASET] '{args.data_dir}' | Found {num_classes} classes: {classes[:8]}{' ...' if num_classes > 8 else ''}", flush=True)
    # Optuna optimisation (optional)
    if args.use_optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, train_loader, val_loader, num_classes, device), n_trials=args.n_trials)
        best_params = study.best_trial.params
        backbone = best_params['backbone']
        lr = best_params['lr']
        weight_decay = best_params['weight_decay']
    else:
        backbone = args.backbone
        lr = args.lr
        weight_decay = args.weight_decay

    model = build_model(num_classes=num_classes, backbone=backbone).to(device)
    if getattr(args, 'resume', None):
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[RESUME] Loading checkpoint from {resume_path}...", flush=True)
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            # Filter out mismatched tensor shapes (e.g. if stem or class count changed)
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"[RESUME] Loaded {len(matched_dict)}/{len(model_dict)} matching parameter tensors from checkpoint.", flush=True)
        else:
            print(f"[WARNING] Checkpoint path {resume_path} not found. Training from scratch.", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Automatic Mixed Precision (AMP) on CUDA
    use_amp = (device.type == 'cuda' and getattr(args, 'amp', True))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("[AMP] Automatic Mixed Precision enabled (Tensor Core acceleration active)", flush=True)

    # MLflow tracking
    # Configure MLflow to use a SQLite backend to avoid file-store maintenance mode
    mlflow_db_path = Path('mlflow.db').resolve()
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.as_posix()}")

    mlflow.set_experiment('image_classification')
    with mlflow.start_run(run_name=f"{backbone}_{num_classes}cls"):
        mlflow.log_params({
            'backbone': backbone,
            'num_classes': num_classes,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'use_amp': use_amp,
        })
        writer = SummaryWriter(log_dir=f'runs/{backbone}')
        model_path = Path('models')
        model_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            epoch_loss = 0.0
            train_bar = tqdm(train_loader, desc=f'[{backbone}] Ep {epoch:02d}/{args.epochs:02d} [Train]', unit='batch', leave=False)
            for inputs, targets in train_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                train_bar.set_postfix(loss=f'{loss.item():.4f}')
            scheduler.step()
            # Validation
            model.eval()
            all_preds, all_labels = [], []
            val_bar = tqdm(val_loader, desc=f'[{backbone}] Ep {epoch:02d}/{args.epochs:02d} [Val]  ', unit='batch', leave=False)
            with torch.no_grad():
                for inputs, targets in val_bar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(targets.cpu())
            y_true = torch.cat(all_labels).numpy()
            y_pred = torch.cat(all_preds).numpy()
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            elapsed = time.time() - start_time
            # Log
            mlflow.log_metric('val_accuracy', acc, step=epoch)
            mlflow.log_metric('val_precision', prec, step=epoch)
            mlflow.log_metric('val_recall', rec, step=epoch)
            mlflow.log_metric('val_f1', f1, step=epoch)
            writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
            writer.add_scalar('Accuracy/val', acc, epoch)

            # Per-epoch checkpointing & memory cleanup to prevent loss of progress or VRAM leaks
            torch.save(model.state_dict(), model_path / f'checkpoint_{backbone}_latest.pth')
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            print(f'[{backbone}] Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}% | Val F1: {f1:.4f} | Time: {elapsed:.1f}s', flush=True)
        # Save artefacts
        timestamp = int(time.time())
        model_path = Path('models')
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save timestamped, backbone-specific, and latest checkpoints
        torch.save(model.state_dict(), model_path / f'model_{backbone}_{timestamp}.pth')
        torch.save(model.state_dict(), model_path / f'model_{backbone}_latest.pth')
        torch.save(model.state_dict(), model_path / 'latest.pth')

        # Save class mappings for dynamic multi-class inference
        with open(model_path / f'classes_{timestamp}.json', 'w') as f:
            json.dump(classes, f, indent=2)
        with open(model_path / 'classes.json', 'w') as f:
            json.dump(classes, f, indent=2)
        mlflow.log_artifact(str(model_path / 'classes.json'))

        # Save backbone metadata
        meta = {
            'backbone': backbone,
            'timestamp': timestamp,
            'num_classes': num_classes,
            'epochs': args.epochs,
            'val_accuracy': acc,
            'val_f1': f1,
            'train_loss': epoch_loss / len(train_loader),
        }
        with open(model_path / f'metadata_{backbone}.json', 'w') as f:
            json.dump(meta, f, indent=2)
        # Export ONNX
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        try:
            torch.onnx.export(model, dummy_input, model_path / f'model_{timestamp}.onnx', opset_version=18)
            mlflow.log_artifact(str(model_path / f'model_{timestamp}.onnx'))
        except Exception as e:
            print(f'Warning: ONNX export skipped: {e}', flush=True)

        # Export TorchScript
        try:
            scripted = torch.jit.trace(model, dummy_input)
            scripted.save(model_path / f'model_{timestamp}.torchscript.pt')
            mlflow.log_artifact(str(model_path / f'model_{timestamp}.torchscript.pt'))
        except Exception as e:
            print(f'Warning: TorchScript export skipped: {e}', flush=True)

        pth_file = model_path / f'model_{backbone}_{timestamp}.pth'
        if pth_file.exists():
            mlflow.log_artifact(str(pth_file))
        print(f'Checkpoints successfully saved to {model_path}/', flush=True)
        return meta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--data-dir', type=str, default='train', help='Root directory of image data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--test-split', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=0, help='Number of DataLoader worker processes (0 for Windows)')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=[
        'resnet101', 'efficientnet_b4', 'mobilenet_v3_large', 'densenet121',
        'convnext_tiny', 'swin_t', 'shufflenet_v2_x1_0', 'resnext50_32x4d',
        'vgg16', 'alexnet', 'vit_small', 'vit_base', 'vit_large',
        'rnn', 'lstm', 'capsnet', 'gnn', 'autoencoder', 'mlp_mixer'
    ])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--use-optuna', action='store_true', help='Enable hyper-parameter optimization')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--amp', action='store_true', default=True, help='Enable Automatic Mixed Precision (default: True)')
    parser.add_argument('--no-amp', dest='amp', action='store_false', help='Disable Automatic Mixed Precision')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume weights from')
    args = parser.parse_args()
    train(args)
