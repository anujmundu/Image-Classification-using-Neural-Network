import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.backbones import build_backbone
from src.ensemble import SoftVotingEnsemble
from src.onnx_engine import ONNXInferenceEngine

# All 11 Official Benchmark Backbones
OFFICIAL_11_BACKBONES = [
    "efficientnet_b4",
    "mobilenet_v3_large",
    "convnext_tiny",
    "densenet121",
    "resnet101",
    "resnext50_32x4d",
    "shufflenet_v2_x1_0",
    "vit_small",
    "mlp_mixer",
    "capsnet",
    "swin_t",
]

@pytest.mark.parametrize("name", OFFICIAL_11_BACKBONES)
def test_all_11_backbones_instantiation_and_forward(name):
    """Ensure all 11 benchmark architectures instantiate and complete a valid forward pass."""
    num_classes = 50
    model = build_backbone(name, num_classes=num_classes)
    assert isinstance(model, nn.Module)
    model.eval()

    # Create batch of 2 dummy images (batch, channels, height, width)
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == dummy.shape[0]
    assert out.shape[1] == num_classes

def test_soft_voting_ensemble_forward():
    """Verify SoftVotingEnsemble probability aggregation on real model checkpoints."""
    ckpt_b4 = REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth"
    ckpt_mob = REPO_ROOT / "models" / "model_mobilenet_v3_large_latest.pth"
    
    if ckpt_b4.exists() and ckpt_mob.exists():
        configs = {
            "efficientnet_b4": ckpt_b4,
            "mobilenet_v3_large": ckpt_mob,
        }
        weights = {"efficientnet_b4": 0.6, "mobilenet_v3_large": 0.4}
        ensemble = SoftVotingEnsemble.from_checkpoints(
            configs, num_classes=50, weights=weights, device=torch.device("cpu")
        )
        
        # Test tensor forward pass
        dummy_tensor = torch.randn(2, 3, 224, 224)
        probs = ensemble.forward(dummy_tensor)
        assert probs.shape == (2, 50)
        # Probabilities must sum to ~1.0 per sample
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-3)
        
        # Test PIL image prediction
        dummy_pil = Image.new("RGB", (224, 224), color=(128, 128, 128))
        class_names = [f"class_{i}" for i in range(50)]
        res = ensemble.predict_image(dummy_pil, class_names)
        assert "prediction" in res
        assert "confidence_pct" in res
        assert len(res["top_rankings"]) == 5

def test_onnx_inference_engine():
    """Verify ONNXInferenceEngine loads exported ONNX model and predicts correctly."""
    onnx_path = REPO_ROOT / "models" / "model_efficientnet_b4_fp32.onnx"
    if onnx_path.exists():
        engine = ONNXInferenceEngine(onnx_path)
        assert engine.session is not None
        
        dummy_pil = Image.new("RGB", (224, 224), color=(200, 100, 50))
        class_names = [f"class_{i}" for i in range(50)]
        res = engine.predict(dummy_pil, class_names)
        assert "prediction" in res
        assert "confidence_pct" in res
        assert len(res["top_rankings"]) == 5
        assert "latency_ms" in res
