import time
import torch
import os
from pathlib import Path

def measure_inference_latency(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device = torch.device('cpu'), runs: int = 10) -> float:
    """Run a few forward passes and return average latency in milliseconds.
    Args:
        model: The model to evaluate (already on ``device``).
        input_tensor: Example input tensor (already on ``device``).
        device: Device for computation.
        runs: Number of repetitions.
    """
    model.eval()
    # Warm‑up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
    end = time.time()
    avg_ms = ((end - start) / runs) * 1000
    return avg_ms

def model_file_size(model_path: Path) -> int:
    """Return file size in bytes of a saved ``.pth`` or ``.onnx`` model."""
    return os.path.getsize(model_path)

def flops_count(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    """Estimate FLOPs using ``fvcore`` if available.
    Returns FLOPs as a float (GigaFLOPs). If ``fvcore`` is not installed, returns -1.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(model, input_tensor)
        flops = analysis.total()
        return flops / 1e9  # convert to GFLOPs
    except Exception:
        return -1.0
