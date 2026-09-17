# demo/streamlit_demo.py
"""Interactive Multi-Model Streamlit Application for 11-Architecture Neural Network Benchmark:
1. EfficientNet-B4 (Compound Scaled CNN)
2. MobileNet-V3-Large (Edge / Mobile CNN)
3. ConvNeXt-Tiny (Modernized Pure-CNN)
4. DenseNet-121 (Dense Feature Concatenation CNN)
5. ResNet-101 (Deep Residual CNN)
6. ResNeXt-50 (Cardinality Residual CNN)
7. ShuffleNet-V2 (Ultra-Lightweight Channel Shuffle CNN)
8. ViT-Small (Vision Transformer)
9. MLP-Mixer (All-MLP Architecture)
10. CapsNet (Capsule Network)
11. Swin-T (Shifted Window ViT)
Plus:
- Soft-Voting Multi-Model Ensemble
- ONNX Runtime Accelerated Inference Engine (FP32 & INT8 Quantized)
"""

import os
import sys
import time
import json
import base64
import tempfile
from pathlib import Path
from PIL import Image
import streamlit as st

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predict import predict, load_model
from src.ensemble import SoftVotingEnsemble
from src.onnx_engine import ONNXInferenceEngine

st.set_page_config(
    page_title="11-Architecture Vision Benchmark Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .paradigm-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .badge-cnn { background-color: #1d4ed8; color: white; }
    .badge-mobile { background-color: #0284c7; color: white; }
    .badge-vit { background-color: #6d28d9; color: white; }
    .badge-mlp { background-color: #047857; color: white; }
    .badge-caps { background-color: #b45309; color: white; }
    .badge-ensemble { background-color: #c026d3; color: white; }
    .badge-onnx { background-color: #ea580c; color: white; }
</style>
""", unsafe_allow_html=True)

# 11 Official Benchmark Models
MODELS_CATALOG = {
    "efficientnet_b4": {
        "name": "EfficientNet-B4",
        "category": "Compound Scaled CNN",
        "badge_class": "badge-cnn",
        "checkpoint": REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth",
        "icon": "🥇",
        "desc": "Deep compound scaling (depth, width, resolution)",
        "accuracy": "90.71%",
        "f1": "0.9072"
    },
    "mobilenet_v3_large": {
        "name": "MobileNet-V3-Large",
        "category": "Mobile / Edge CNN",
        "badge_class": "badge-mobile",
        "checkpoint": REPO_ROOT / "models" / "model_mobilenet_v3_large_latest.pth",
        "icon": "🥈",
        "desc": "Inverted residuals with hard-swish and SE attention",
        "accuracy": "83.50%",
        "f1": "0.8326"
    },
    "convnext_tiny": {
        "name": "ConvNeXt-Tiny",
        "category": "Modernized Pure-CNN",
        "badge_class": "badge-cnn",
        "checkpoint": REPO_ROOT / "models" / "model_convnext_tiny_latest.pth",
        "icon": "🥉",
        "desc": "Inverted bottleneck with 7x7 depthwise convolutions",
        "accuracy": "82.52%",
        "f1": "0.8253"
    },
    "densenet121": {
        "name": "DenseNet-121",
        "category": "Dense Connectivity CNN",
        "badge_class": "badge-cnn",
        "checkpoint": REPO_ROOT / "models" / "model_densenet121_latest.pth",
        "icon": "🔗",
        "desc": "Direct feature concatenation across 121 layers",
        "accuracy": "81.28%",
        "f1": "0.8140"
    },
    "resnet101": {
        "name": "ResNet-101",
        "category": "Deep Residual CNN",
        "badge_class": "badge-cnn",
        "checkpoint": REPO_ROOT / "models" / "model_resnet101_latest.pth",
        "icon": "🧱",
        "desc": "Classical 101-layer identity shortcut skip connections",
        "accuracy": "81.28%",
        "f1": "0.8099"
    },
    "resnext50_32x4d": {
        "name": "ResNeXt-50 (32x4d)",
        "category": "Cardinality Residual CNN",
        "badge_class": "badge-cnn",
        "checkpoint": REPO_ROOT / "models" / "model_resnext50_32x4d_latest.pth",
        "icon": "🔀",
        "desc": "32 grouped multi-branch aggregated transformations",
        "accuracy": "80.58%",
        "f1": "0.8031"
    },
    "shufflenet_v2_x1_0": {
        "name": "ShuffleNet-V2 (1.0x)",
        "category": "Channel Shuffle Edge CNN",
        "badge_class": "badge-mobile",
        "checkpoint": REPO_ROOT / "models" / "model_shufflenet_v2_x1_0_latest.pth",
        "icon": "⚡",
        "desc": "Channel split and shuffle with 5.4MB footprint",
        "accuracy": "79.47%",
        "f1": "0.7946"
    },
    "vit_small": {
        "name": "ViT-Small",
        "category": "Vision Transformer (ViT)",
        "badge_class": "badge-vit",
        "checkpoint": REPO_ROOT / "models" / "model_vit_small_latest.pth",
        "icon": "👁️",
        "desc": "Patch projection with multi-head self-attention",
        "accuracy": "38.28%",
        "f1": "0.3802"
    },
    "mlp_mixer": {
        "name": "MLP-Mixer",
        "category": "All-MLP Architecture",
        "badge_class": "badge-mlp",
        "checkpoint": REPO_ROOT / "models" / "model_mlp_mixer_latest.pth",
        "icon": "🧪",
        "desc": "Alternating spatial token and channel perceptron mixing",
        "accuracy": "27.88%",
        "f1": "0.2663"
    },
    "capsnet": {
        "name": "CapsNet",
        "category": "Vector Capsules",
        "badge_class": "badge-caps",
        "checkpoint": REPO_ROOT / "models" / "model_capsnet_latest.pth",
        "icon": "💊",
        "desc": "Vector capsule activations with dynamic routing",
        "accuracy": "22.61%",
        "f1": "0.1994"
    },
    "swin_t": {
        "name": "Swin-T",
        "category": "Shifted Window ViT",
        "badge_class": "badge-vit",
        "checkpoint": REPO_ROOT / "models" / "model_swin_t_latest.pth",
        "icon": "🪟",
        "desc": "Hierarchical shifted-window local self-attention",
        "accuracy": "5.96%",
        "f1": "0.0236"
    }
}

# Load registered classes
classes_file = REPO_ROOT / "models" / "classes.json"
if classes_file.exists():
    try:
        with open(classes_file, "r") as f:
            CLASS_NAMES = json.load(f)
    except Exception:
        CLASS_NAMES = [f"class_{i}" for i in range(50)]
else:
    CLASS_NAMES = [f"class_{i}" for i in range(50)]

# Cache models in memory for instant inference
@st.cache_resource
def get_cached_model(backbone: str, num_classes: int):
    if backbone in MODELS_CATALOG:
        ckpt_path = MODELS_CATALOG[backbone]["checkpoint"]
        if ckpt_path.exists():
            return load_model(ckpt_path, num_classes=num_classes, backbone=backbone)
    return None

@st.cache_resource
def get_cached_ensemble():
    configs = {
        "efficientnet_b4": REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth",
        "mobilenet_v3_large": REPO_ROOT / "models" / "model_mobilenet_v3_large_latest.pth",
        "convnext_tiny": REPO_ROOT / "models" / "model_convnext_tiny_latest.pth",
        "resnet101": REPO_ROOT / "models" / "model_resnet101_latest.pth",
    }
    available = {k: v for k, v in configs.items() if v.exists()}
    weights = {"efficientnet_b4": 0.40, "mobilenet_v3_large": 0.25, "convnext_tiny": 0.20, "resnet101": 0.15}
    if available:
        return SoftVotingEnsemble.from_checkpoints(available, num_classes=len(CLASS_NAMES), weights=weights)
    return None

@st.cache_resource
def get_cached_onnx_engine(quantized: bool = False):
    model_name = "model_efficientnet_b4_int8.onnx" if quantized else "model_efficientnet_b4_fp32.onnx"
    onnx_path = REPO_ROOT / "models" / model_name
    if onnx_path.exists():
        return ONNXInferenceEngine(onnx_path)
    return None

# Sidebar
st.sidebar.markdown("## 🧠 Benchmark Suite")
st.sidebar.markdown("**Dataset:** Caltech-50 Curated Benchmark")
st.sidebar.markdown(f"**Total Classes:** `{len(CLASS_NAMES)}`")
st.sidebar.markdown(f"**Total Architectures:** `11`")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🏆 Top Contenders")
for k in ["efficientnet_b4", "mobilenet_v3_large", "convnext_tiny", "densenet121", "resnet101"]:
    v = MODELS_CATALOG[k]
    exists = v["checkpoint"].exists()
    status = "✅" if exists else "⚠️"
    st.sidebar.markdown(f"{status} **{v['name']}**: `{v['accuracy']}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Advanced Engines")
ens_ready = "✅ Ready (98.92% Top-5)" if (REPO_ROOT / "models" / "model_efficientnet_b4_latest.pth").exists() else "⚠️ Missing"
st.sidebar.markdown(f"**Soft-Voting Ensemble**: {ens_ready}")

onnx_fp32_ready = "✅ Ready (18.9 ms)" if (REPO_ROOT / "models" / "model_efficientnet_b4_fp32.onnx").exists() else "⚠️ Missing"
st.sidebar.markdown(f"**ONNX Runtime (FP32)**: {onnx_fp32_ready}")

onnx_int8_ready = "✅ Ready (17.5 MB)" if (REPO_ROOT / "models" / "model_efficientnet_b4_int8.onnx").exists() else "⚠️ Missing"
st.sidebar.markdown(f"**ONNX INT8 Quantized**: {onnx_int8_ready}")

st.sidebar.markdown("---")
st.sidebar.caption("Image Classification using Neural Network | 11 Architectures Benchmark")

# App Header
st.markdown('<div class="main-header">🧠 11-Architecture Neural Network Vision Benchmark</div>', unsafe_allow_html=True)
st.caption("Comprehensive comparative evaluation across 5 deep learning paradigms: CNNs, Modern Pure-ConvNets, Vision Transformers, All-MLP, and Capsule Networks.")

# Navigation Tabs
tab_arena, tab_single, tab_ensemble, tab_leaderboard = st.tabs([
    "🥊 Multi-Model Comparison Arena",
    "🔬 Single Architecture Deep Dive",
    "🤝 Soft-Voting Ensemble & ONNX Engine",
    "📊 Benchmark Leaderboard & Visualizations"
])

# Sample images helper from sample_images/ directory
sample_dir = REPO_ROOT / "sample_images"
sample_images = sorted(list(sample_dir.glob("sample_*.jpg")) + list(sample_dir.glob("sample_*.png")))

# -----------------------------------------------------------------------------
# TAB 1: MULTI-MODEL COMPARISON ARENA
# -----------------------------------------------------------------------------
with tab_arena:
    st.subheader("Benchmark Architectures Side-by-Side on the Same Input Image")
    
    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        arena_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="arena_upload")
    with col_sample:
        selected_sample = None
        if sample_images:
            sample_names = ["(None)"] + [p.name for p in sample_images]
            chosen = st.selectbox("Or choose a benchmark test sample:", sample_names, key="arena_sample")
            if chosen != "(None)":
                selected_sample = next(p for p in sample_images if p.name == chosen)

    active_image = None
    if arena_file is not None:
        active_image = Image.open(arena_file).convert("RGB")
    elif selected_sample is not None:
        active_image = Image.open(selected_sample).convert("RGB")

    default_models = ["efficientnet_b4", "mobilenet_v3_large", "convnext_tiny", "vit_small"]
    selected_arena_models = st.multiselect(
        "Choose models to compare in the Arena:",
        options=list(MODELS_CATALOG.keys()),
        default=default_models,
        format_func=lambda k: f"{MODELS_CATALOG[k]['icon']} {MODELS_CATALOG[k]['name']} ({MODELS_CATALOG[k]['category']})"
    )

    if active_image is not None and selected_arena_models:
        tmp_path = Path(tempfile.gettempdir()) / "arena_test_image.png"
        active_image.save(tmp_path)

        st.markdown("---")
        st.image(active_image, caption="Query Benchmark Input Image", width=280)

        with st.spinner(f"⚡ Running real-time inference across {len(selected_arena_models)} architectures..."):
            cols = st.columns(len(selected_arena_models))

            for idx, backbone in enumerate(selected_arena_models):
                info = MODELS_CATALOG[backbone]
                with cols[idx]:
                    st.markdown(f'<span class="paradigm-badge {info["badge_class"]}">{info["category"]}</span>', unsafe_allow_html=True)
                    st.markdown(f"#### {info['icon']} {info['name']}")

                    if not info["checkpoint"].exists():
                        st.warning("Checkpoint not found.")
                        continue

                    m = get_cached_model(backbone, len(CLASS_NAMES))
                    start_t = time.perf_counter()
                    res = predict(tmp_path, info["checkpoint"], CLASS_NAMES, backbone=backbone, loaded_model=m)
                    lat_ms = (time.perf_counter() - start_t) * 1000.0

                    conf_pct = res["confidence"] * 100
                    st.success(f"**{res['label']}**  \nConfidence: **{conf_pct:.2f}%**")
                    st.caption(f"⏱️ Latency: **{lat_ms:.1f} ms**")

                    if res.get("top5"):
                        st.markdown("**Top-3 Predictions:**")
                        for t in res["top5"][:3]:
                            st.write(f"• `{t['label']}`: {t['confidence']*100:.1f}%")

                    if res.get("grad_cam"):
                        cam_bytes = base64.b64decode(res["grad_cam"])
                        st.image(cam_bytes, caption=f"Grad-CAM Heatmap", use_container_width=True)

        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    else:
        st.info("👆 Upload an image or select a sample image above to benchmark architectures side-by-side!")

# -----------------------------------------------------------------------------
# TAB 2: SINGLE MODEL DEEP DIVE
# -----------------------------------------------------------------------------
with tab_single:
    st.subheader("In-Depth Inspection & Explainability for Any Architecture")
    single_backbone = st.selectbox(
        "Select Architecture Paradigm",
        list(MODELS_CATALOG.keys()),
        format_func=lambda k: f"{MODELS_CATALOG[k]['icon']} {MODELS_CATALOG[k]['name']} — {MODELS_CATALOG[k]['category']} (Top-1: {MODELS_CATALOG[k]['accuracy']})"
    )
    chosen_info = MODELS_CATALOG[single_backbone]

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        s_file = st.file_uploader("Upload Image for Single Inspection", type=["jpg", "jpeg", "png"], key="single_upload")
        s_sample = None
        if sample_images:
            s_chosen = st.selectbox("Or choose sample image:", ["(None)"] + [p.name for p in sample_images], key="single_sample")
            if s_chosen != "(None)":
                s_sample = next(p for p in sample_images if p.name == s_chosen)

        active_s_image = None
        if s_file is not None:
            active_s_image = Image.open(s_file).convert("RGB")
        elif s_sample is not None:
            active_s_image = Image.open(s_sample).convert("RGB")

        if active_s_image is not None:
            st.image(active_s_image, caption="Query Input Image", use_container_width=True)

    with col_s2:
        if active_s_image is not None and chosen_info["checkpoint"].exists():
            tmp_s = Path(tempfile.gettempdir()) / "single_test_image.png"
            active_s_image.save(tmp_s)

            with st.spinner(f"Running inference on {chosen_info['name']}..."):
                m = get_cached_model(single_backbone, len(CLASS_NAMES))
                start_s = time.perf_counter()
                res = predict(tmp_s, chosen_info["checkpoint"], CLASS_NAMES, backbone=single_backbone, loaded_model=m)
                s_lat = (time.perf_counter() - start_s) * 1000.0

            st.success(f"### Top Prediction: **{res['label']}** ({res['confidence']*100:.2f}%)")
            st.caption(f"Architecture: **{chosen_info['name']}** | Benchmark Accuracy: **{chosen_info['accuracy']}** | Latency: **{s_lat:.1f} ms**")

            if res.get("top5"):
                st.markdown("#### Top-5 Class Probability Distribution")
                for item in res["top5"]:
                    pct = item["confidence"] * 100
                    st.write(f"**{item['label']}**: {pct:.2f}%")
                    st.progress(min(1.0, float(item["confidence"])))

            if res.get("grad_cam"):
                st.markdown("#### 🔍 Visual Explainability Heatmap")
                cam_bytes = base64.b64decode(res["grad_cam"])
                st.image(cam_bytes, caption=f"Explainability Overlay ({chosen_info['name']})", use_container_width=True)

            if tmp_s.exists():
                try:
                    tmp_s.unlink()
                except Exception:
                    pass

# -----------------------------------------------------------------------------
# TAB 3: SOFT-VOTING ENSEMBLE & ONNX ENGINE
# -----------------------------------------------------------------------------
with tab_ensemble:
    st.subheader("🤝 Advanced Production Deployment: Multi-Model Ensemble & ONNX Runtime")
    
    col_ens1, col_ens2 = st.columns([1, 1])
    
    with col_ens1:
        st.markdown("### 🏆 Soft-Voting Ensemble (Option B)")
        st.markdown("""
        Fuses predictions across top-performing heterogeneous architectures:
        - **EfficientNet-B4** (Weight: 40%)
        - **MobileNet-V3-Large** (Weight: 25%)
        - **ConvNeXt-Tiny** (Weight: 20%)
        - **ResNet-101** (Weight: 15%)
        
        **Benchmark Yield:**
        - Top-5 Accuracy: **`98.92%`** (Outperforms all individual standalone models)
        - Top-1 Accuracy: **`90.66%`**
        """)
        
        ens = get_cached_ensemble()
        if ens:
            st.success("✅ Soft-Voting Ensemble is loaded and active in memory.")
        else:
            st.warning("⚠️ Some ensemble checkpoints are missing in models/.")

    with col_ens2:
        st.markdown("### ⚡ ONNX Runtime High-Speed Engine (Option C)")
        st.markdown("""
        Hardware-accelerated graph execution with post-training INT8 quantization:
        - **ONNX FP32 Engine:** **`18.92 ms`** per image (**3.01x faster** than PyTorch CPU)
        - **ONNX INT8 Quantized:** **`17.56 MB`** footprint (**74.2% disk & memory reduction**)
        """)
        
        onnx_engine = get_cached_onnx_engine(quantized=False)
        onnx_int8_engine = get_cached_onnx_engine(quantized=True)
        if onnx_engine:
            st.success("✅ ONNX FP32 Runtime Engine ready.")
        if onnx_int8_engine:
            st.success("✅ ONNX INT8 Quantized Engine ready.")

    st.markdown("---")
    st.subheader("Interactive Evaluation on Ensemble & ONNX")
    
    ens_test_img = st.file_uploader("Upload Image for Ensemble / ONNX Evaluation", type=["jpg", "jpeg", "png"], key="ens_upload")
    chosen_ens_sample = None
    if sample_images:
        ch = st.selectbox("Or choose sample image:", ["(None)"] + [p.name for p in sample_images], key="ens_sample")
        if ch != "(None)":
            chosen_ens_sample = next(p for p in sample_images if p.name == ch)

    active_ens_img = None
    if ens_test_img:
        active_ens_img = Image.open(ens_test_img).convert("RGB")
    elif chosen_ens_sample:
        active_ens_img = Image.open(chosen_ens_sample).convert("RGB")

    if active_ens_img:
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.image(active_ens_img, caption="Query Input Image", width=300)
        
        with col_res2:
            st.markdown("#### Comparison of Engine Outputs:")
            
            # 1. Soft-Voting Ensemble prediction
            if ens:
                start_ens = time.perf_counter()
                res_ens = ens.predict_image(active_ens_img, CLASS_NAMES)
                ens_lat = (time.perf_counter() - start_ens) * 1000.0
                st.markdown(f"**🤝 Soft-Voting Ensemble:** `{res_ens['prediction']}` ({res_ens['confidence_pct']}%) | Latency: `{ens_lat:.1f} ms`")
                for t in res_ens.get("top_rankings", [])[:3]:
                    st.caption(f"&nbsp;&nbsp;• Rank #{t['rank']} `{t['class']}`: {t['confidence_pct']}%")
            
            # 2. ONNX FP32 prediction
            if onnx_engine:
                start_onnx = time.perf_counter()
                res_onnx = onnx_engine.predict(active_ens_img, CLASS_NAMES)
                onnx_lat = (time.perf_counter() - start_onnx) * 1000.0
                st.markdown(f"**⚡ ONNX Runtime (FP32):** `{res_onnx['prediction']}` ({res_onnx['confidence_pct']}%) | Latency: `{onnx_lat:.1f} ms`")
            
            # 3. ONNX INT8 prediction
            if onnx_int8_engine:
                start_int8 = time.perf_counter()
                res_int8 = onnx_int8_engine.predict(active_ens_img, CLASS_NAMES)
                int8_lat = (time.perf_counter() - start_int8) * 1000.0
                st.markdown(f"**💾 ONNX INT8 Quantized:** `{res_int8['prediction']}` ({res_int8['confidence_pct']}%) | Latency: `{int8_lat:.1f} ms`")

# -----------------------------------------------------------------------------
# TAB 4: BENCHMARK LEADERBOARD & VISUALIZATIONS
# -----------------------------------------------------------------------------
with tab_leaderboard:
    st.subheader("🏆 Official 11-Architecture Benchmark Leaderboard & Analytics")
    
    # Leaderboard KPI Summary
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🥇 Champion Top-1 Acc", "90.71%", "EfficientNet-B4")
    k2.metric("🤝 Ensemble Top-5 Acc", "98.92%", "Soft-Voting Fusion")
    k3.metric("⚡ Fastest Training", "34.1s / epoch", "ShuffleNet-V2")
    k4.metric("🚀 ONNX Acceleration", "3.01x Speedup", "18.9 ms / image")

    st.markdown("---")
    st.markdown("### Official 11-Model Performance Ranking")
    
    leaderboard_table = [
        {"Rank": "🥇 #1", "Paradigm": "Compound Scaled CNN", "Model": "EfficientNet-B4", "Backbone": "efficientnet_b4", "Val Acc (Top-1)": "90.71%", "Val F1": "0.9072", "Train Loss": "0.0076", "Traits": "Compound scaling (depth, width, resolution)"},
        {"Rank": "🥈 #2", "Paradigm": "Mobile / Edge CNN", "Model": "MobileNet-V3-Large", "Backbone": "mobilenet_v3_large", "Val Acc (Top-1)": "83.50%", "Val F1": "0.8326", "Train Loss": "0.0053", "Traits": "Inverted residuals with hard-swish & SE"},
        {"Rank": "🥉 #3", "Paradigm": "Modernized Pure-CNN", "Model": "ConvNeXt-Tiny", "Backbone": "convnext_tiny", "Val Acc (Top-1)": "82.52%", "Val F1": "0.8253", "Train Loss": "0.0039", "Traits": "Inverted bottleneck with 7x7 depthwise"},
        {"Rank": "#4", "Paradigm": "Dense Connectivity", "Model": "DenseNet-121", "Backbone": "densenet121", "Val Acc (Top-1)": "81.28%", "Val F1": "0.8140", "Train Loss": "0.0113", "Traits": "Direct feature concatenation across 121 layers"},
        {"Rank": "#5", "Paradigm": "Deep Residual CNN", "Model": "ResNet-101", "Backbone": "resnet101", "Val Acc (Top-1)": "81.28%", "Val F1": "0.8099", "Train Loss": "0.0053", "Traits": "Classical 101-layer identity shortcut skips"},
        {"Rank": "#6", "Paradigm": "Cardinality Residual", "Model": "ResNeXt-50 (32x4d)", "Backbone": "resnext50_32x4d", "Val Acc (Top-1)": "80.58%", "Val F1": "0.8031", "Train Loss": "0.0068", "Traits": "32 grouped multi-branch transformations"},
        {"Rank": "#7", "Paradigm": "Channel Shuffle CNN", "Model": "ShuffleNet-V2 (1.0x)", "Backbone": "shufflenet_v2_x1_0", "Val Acc (Top-1)": "79.47%", "Val F1": "0.7946", "Train Loss": "0.0327", "Traits": "Channel split and shuffle (5.4MB footprint)"},
        {"Rank": "#8", "Paradigm": "Vision Transformer", "Model": "ViT-Small", "Backbone": "vit_small", "Val Acc (Top-1)": "38.28%", "Val F1": "0.3802", "Train Loss": "0.8645", "Traits": "Patch projection with multi-head self-attention"},
        {"Rank": "#9", "Paradigm": "All-MLP Architecture", "Model": "MLP-Mixer", "Backbone": "mlp_mixer", "Val Acc (Top-1)": "27.88%", "Val F1": "0.2663", "Train Loss": "1.8723", "Traits": "Alternating spatial token & channel MLP mixing"},
        {"Rank": "#10", "Paradigm": "Vector Capsules", "Model": "CapsNet", "Backbone": "capsnet", "Val Acc (Top-1)": "22.61%", "Val F1": "0.1994", "Train Loss": "2.8182", "Traits": "Vector capsule activations with dynamic routing"},
        {"Rank": "#11", "Paradigm": "Shifted Window ViT", "Model": "Swin-T", "Backbone": "swin_t", "Val Acc (Top-1)": "5.96%", "Val F1": "0.0236", "Train Loss": "3.7014", "Traits": "Hierarchical shifted-window local self-attention"}
    ]
    st.dataframe(leaderboard_table, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 High-Resolution 11-Model Visualizations Gallery (300 DPI)")
    
    vis_options = {
        "Leaderboard Bar Chart": REPO_ROOT / "Result" / "All_11_Models_Accuracy_Leaderboard_BarChart.png",
        "Executive Summary Infographic": REPO_ROOT / "Result" / "Grand_11_Architecture_Summary_Infographic.png",
        "Accuracy Convergence Curves": REPO_ROOT / "Result" / "Training_And_Validation_Accuracy_Comparison_11_Models.png",
        "Loss Convergence Curves": REPO_ROOT / "Result" / "Training_Loss_Comparison_11_Models.png",
        "Throughput vs Accuracy Pareto Frontier": REPO_ROOT / "Result" / "Throughput_vs_Accuracy_Pareto_Frontier_11_Models.png",
        "Parameter Efficiency Metric": REPO_ROOT / "Result" / "Parameter_Efficiency_Metric_11_Models.png",
        "Disk Footprint Comparison": REPO_ROOT / "Result" / "Model_Checkpoint_Disk_Footprint_11_Models.png",
        "F1 vs Accuracy Calibration": REPO_ROOT / "Result" / "Macro_F1_vs_Accuracy_Correlation_11_Models.png",
        "Ensemble Lift Comparison": REPO_ROOT / "Result" / "Ensemble_vs_Individual_Models_Comparison.png",
        "ONNX & INT8 Quantization": REPO_ROOT / "Result" / "ONNX_and_INT8_Quantization_Benchmark.png"
    }

    selected_vis = st.selectbox("Select Benchmark Visualization to Inspect:", list(vis_options.keys()))
    chart_path = vis_options[selected_vis]
    if chart_path.exists():
        st.image(str(chart_path), caption=f"Publication-Grade Visualization: {selected_vis}", use_container_width=True)
    else:
        st.warning(f"Visualization not found at {chart_path.name}.")
