# src/onnx_engine.py
"""High-Performance Inference Engine utilizing ONNX Runtime and Quantized INT8 Models."""

import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import onnxruntime as ort

class ONNXInferenceEngine:
    """Production ONNX Runtime inference engine supporting FP32 and INT8 quantized models."""

    def __init__(self, model_path: Path, providers: Optional[List[str]] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
            
        # Prioritize CPU or CUDA provider based on availability
        available_providers = ort.get_available_providers()
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
            
        # Session options for multi-threaded high-throughput execution
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Resize and normalize PIL image to NCHW float32 numpy tensor."""
        img = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)
        arr = np.transpose(arr, (2, 0, 1))  # (3, 224, 224)
        arr = np.expand_dims(arr, axis=0)   # (1, 3, 224, 224)
        arr = (arr - self.mean) / self.std
        return arr.astype(np.float32)

    def predict(self, image: Image.Image, class_names: List[str], top_k: int = 5) -> Dict:
        """Run accelerated ONNX inference and return Top-K predictions with latency."""
        input_tensor = self.preprocess_image(image)
        
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        logits = outputs[0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top_indices = np.argsort(probs)[::-1][:min(top_k, len(class_names))]
        
        top1_idx = top_indices[0]
        top1_class = class_names[top1_idx]
        top1_conf = float(probs[top1_idx] * 100.0)
        
        rankings = [
            {"rank": i + 1, "class": class_names[idx], "confidence_pct": round(float(probs[idx] * 100.0), 2)}
            for i, idx in enumerate(top_indices)
        ]
        
        return {
            "prediction": top1_class,
            "confidence_pct": round(top1_conf, 2),
            "latency_ms": round(latency_ms, 2),
            "throughput_fps": round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0,
            "top_rankings": rankings,
            "engine": "ONNX Runtime",
            "model": self.model_path.name
        }

    def benchmark(self, iterations: int = 100) -> Tuple[float, float]:
        """Benchmark inference latency (ms) and throughput (FPS) over N iterations."""
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = self.session.run([self.output_name], {self.input_name: dummy})
            
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.session.run([self.output_name], {self.input_name: dummy})
        total_time = time.perf_counter() - t0
        
        latency_ms = (total_time / iterations) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        return latency_ms, fps
