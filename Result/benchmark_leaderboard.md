# 🏆 11-Architecture Deep Learning Benchmark Leaderboard

**Dataset:** Caltech-256 (50 Curated Object Classes)  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (Tensor Core Mixed Precision AMP Active)  
**Evaluation:** 20 Epochs convergence per architecture  

| Rank | Architectural Paradigm | Model Architecture | Backbone Flag | Val Accuracy (Top-1) | Val F1 (Weighted) | Final Train Loss | Primary Trait |
|---|---|---|---|---|---|---|---|
| **#1** | **CNN (Compound Scaling)** | **EfficientNet-B4** | `efficientnet_b4` | **90.71%** | **0.9072** | 0.0076 | 🥇 Global Accuracy Champion |
| **#2** | **Mobile / Edge CNN** | **MobileNet-V3-Large** | `mobilenet_v3_large` | **83.50%** | **0.8326** | 0.0053 | ⚡ Ultra-fast Edge Inference |
| **#3** | **Modernized Pure-CNN** | **ConvNeXt-Tiny** | `convnext_tiny` | **82.52%** | **0.8253** | 0.0039 | 🚀 Modern 7x7 Depthwise Conv |
| **#4** | **Dense Connectivity** | **DenseNet-121** | `densenet121` | **81.28%** | **0.8140** | 0.0113 | 🔗 Direct Layer Feature Reuse |
| **#5** | **Deep Residual CNN** | **ResNet-101** | `resnet101` | **81.28%** | **0.8099** | 0.0053 | 🏛️ Classical Residual Benchmark |
| **#6** | **Cardinality Residual** | **ResNeXt-50 (32x4d)** | `resnext50_32x4d` | **80.58%** | **0.8031** | 0.0068 | 🔀 Multi-Branch Aggregation |
| **#7** | **Channel Shuffle CNN** | **ShuffleNet-V2 (1.0x)** | `shufflenet_v2_x1_0` | **79.47%** | **0.7946** | 0.0327 | 📱 Embedded Device Specialist |
| **#8** | **Vision Transformer** | **ViT-Small (Patch 16)** | `vit_small` | **38.28%** | **0.3802** | 0.8645 | 👁️ Global Multi-Head Self-Attention |
| **#9** | **All-MLP Architecture** | **MLP-Mixer** | `mlp_mixer` | **27.88%** | **0.2663** | 1.8723 | 🎛️ Alternating Token/Channel Mixing |
| **#10** | **Vector Capsules** | **Capsule Network** | `capsnet` | **22.61%** | **0.1994** | 2.8182 | 🧭 Part-Whole Spatial Pose Routing |
| **#11** | **Shifted Window ViT** | **Swin-T** | `swin_t` | **5.96%** | **0.0236** | 3.7014 | 🪟 Hierarchical Local Attention |

---

## 🔬 Architectural Takeaways & Empirical Analysis

1. **Top Tier (80% - 90% Accuracy): The ConvNet Renaissance**
   - **EfficientNet-B4** remains the undisputed benchmark champion (**90.71%**), leveraging compound scaling of depth, width, and resolution.
   - **MobileNet-V3-Large** (**83.50%**) and **ConvNeXt-Tiny** (**82.52%**) prove that modern convolutional techniques (hard-swish, inverted residuals, $7\times 7$ depthwise kernels) offer incredible accuracy-to-parameter ratios.
   - **DenseNet-121** (**81.28%**) and **ResNet-101** (**81.28%**) demonstrate the enduring stability of skip connections and feature concatenation.

2. **Mid Tier (79% - 80% Accuracy): Ultra-Compact Mobile Backbones**
   - **ShuffleNet-V2** achieved **79.47%** accuracy in just **34 seconds per epoch**, making it the absolute fastest model to train and deploy.
   - **ResNeXt-50** reached **80.58%** by expanding cardinality (32 grouped branches) instead of raw depth.

3. **Transformers & Inductive Bias on 50-Class Curated Datasets:**
   - Pure Vision Transformers (**ViT-Small** at 38.28% and **Swin-T** at 5.96%) lack the translational equivariance and local inductive bias inherent to CNNs. On medium-sized datasets without hundreds of epochs of heavy pre-training augmentations (like RandAugment / Mixup), ConvNets outperform ViTs by a wide margin.