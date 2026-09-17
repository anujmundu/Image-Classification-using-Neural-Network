# src/ensemble.py
"""Multi-Model Soft Voting Ensemble Architecture:
Fuses predicted class probabilities across complementary deep learning paradigms:
  P_ensemble(c|x) = sum_{i=1}^k w_i * P_i(c|x)
where sum(w_i) = 1.0.
"""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.predict import load_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SoftVotingEnsemble(nn.Module):
    """Ensemble of neural network models using weighted soft probability voting."""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None, model_names: Optional[List[str]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def from_checkpoints(
        cls,
        model_configs: Dict[str, Path],
        num_classes: int = 50,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None
    ) -> "SoftVotingEnsemble":
        """Factory constructor loading models directly from checkpoint file paths."""
        target_device = device or DEVICE
        loaded_models = []
        names = []
        weight_list = []

        for backbone, ckpt_path in model_configs.items():
            ckpt = Path(ckpt_path)
            if ckpt.exists():
                m = load_model(ckpt, num_classes=num_classes, backbone=backbone)
                m.to(target_device)
                m.eval()
                loaded_models.append(m)
                names.append(backbone)
                w = weights.get(backbone, 1.0) if weights else 1.0
                weight_list.append(w)

        if not loaded_models:
            raise ValueError("No valid checkpoints could be loaded for ensemble.")

        return cls(models=loaded_models, weights=weight_list, model_names=names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing weighted average of softmax probabilities.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
        Returns:
            ensemble_probs: Tensor of shape (B, num_classes).
        """
        ensemble_probs = None
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            dev = next(model.parameters()).device
            with torch.no_grad():
                logits = model(x.to(dev))
                probs = F.softmax(logits, dim=1).to(x.device)
                
            weighted_probs = probs * weight
            if ensemble_probs is None:
                ensemble_probs = weighted_probs
            else:
                ensemble_probs += weighted_probs
                
        return ensemble_probs

    def predict_image(self, image_input: Union[Path, str, Image.Image], class_names: List[str], top_k: int = 5) -> Dict:
        """Predict class probabilities for a single image file or PIL Image."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        tensor = self.preprocess(img).unsqueeze(0)
        dev = next(self.models[0].parameters()).device if len(self.models) > 0 else DEVICE
        tensor = tensor.to(dev)
        
        probs = self.forward(tensor)[0]
        top_probs, top_indices = torch.topk(probs, min(top_k, len(class_names)))
        
        top1_idx = top_indices[0].item()
        top1_class = class_names[top1_idx]
        top1_conf = top_probs[0].item() * 100.0
        
        rankings = [
            {"rank": i + 1, "class": class_names[idx.item()], "confidence_pct": round(prob.item() * 100.0, 2)}
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
        ]
        
        return {
            "prediction": top1_class,
            "confidence_pct": round(top1_conf, 2),
            "top_rankings": rankings,
            "weights_used": dict(zip(self.model_names, [round(w, 3) for w in self.weights]))
        }
