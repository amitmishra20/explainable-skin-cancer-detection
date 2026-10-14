"""
Explainable Skin Cancer Detection Using CNN
Academic Research & Demonstration Dashboard
Architecture: MobileNetV2 + Grad-CAM Explainability | Dataset: HAM10000 (7 Classes)
"""

import os
import sys
import json
import glob
import io
import hashlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="Explainable Skin Cancer AI — Research Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure local imports work cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset import (
    CANONICAL_CLASSES,
    CLASS_METADATA,
    create_demo_samples
)

# Paths
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "skin_lesion_model.keras")
CLASSES_PATH = os.path.join(CURRENT_DIR, "models", "classes.json")
CONFIG_PATH = os.path.join(CURRENT_DIR, "models", "model_config.json")
EVAL_PATH = os.path.join(CURRENT_DIR, "results", "evaluation_report.json")
if not os.path.exists(EVAL_PATH):
    EVAL_PATH = os.path.join(CURRENT_DIR, "models", "evaluation_report.json")

CM_PATH = os.path.join(CURRENT_DIR, "results", "confusion_matrix.png")
if not os.path.exists(CM_PATH):
    CM_PATH = os.path.join(CURRENT_DIR, "models", "confusion_matrix.png")

SAMPLE_DIR = os.path.join(CURRENT_DIR, "data", "sample_images")


# Custom CSS for Dark Graphite Aesthetic
CUSTOM_CSS = """
<style>
    /* Dark Graphite Base Theme */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .hero-container {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #f0f6fc;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #8b949e;
        font-size: 1.05rem;
        margin: 0 0 14px 0;
    }
    .badge-disclaimer {
        display: inline-block;
        background-color: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Metric & Card styling */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value-high {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ef4444;
        margin: 4px 0;
    }
    .metric-value-benign {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
        margin: 4px 0;
    }
    .metric-value-neutral {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 4px 0;
    }

    /* Image Display Container */
    .image-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    /* Explanatory callout */
    .callout-explain {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 14px;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .callout-warning {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 14px 18px;
        color: #fca5a5;
        font-size: 0.88rem;
        margin-top: 20px;
    }

    /* Tab styling overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px 8px 0 0;
        color: #8b949e;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f0f6fc !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading deep learning model weights...")
def get_model():
    """Caches model in memory to prevent reload overhead on every user interaction."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load neural network model: {e}")
        return None


def preprocess_pil(image: Image.Image, target_size=(224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses PIL image to MobileNetV2 input standards."""
    import tensorflow as tf
    rgb_img = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(rgb_img, dtype=np.float32)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    tensor = np.expand_dims(preprocessed, axis=0)
    return rgb_img, tensor


def render_header():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Explainable Skin Cancer Detection Using CNN</div>
        <div class="hero-sub">Deep Transfer Learning (MobileNetV2) &amp; Visual Attribution (Grad-CAM) on HAM10000</div>
        <div class="badge-disclaimer">Academic Research Demonstration Prototype</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    render_header()

    # Load classes
    classes = CANONICAL_CLASSES
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, "r") as f:
                classes = json.load(f)
        except Exception:
            classes = CANONICAL_CLASSES

    # Sidebar Navigation & Settings
    with st.sidebar:
        st.subheader("Configuration & Controls")
        
        # Grad-CAM Visualization options
        st.markdown("**Grad-CAM Settings**")
        colormap = st.selectbox("Colormap", ["jet", "turbo", "inferno", "magma", "plasma"], index=0)
        overlay_alpha = st.slider("Heatmap Blending Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
        
        st.divider()
        st.markdown("**Model Details**")
        st.caption("• Backbone: MobileNetV2 (ImageNet)")
        st.caption("• Target Layer: out_relu (Conv2D)")
        st.caption("• Input Resolution: 224 × 224 px")
        st.caption("• Categories: 7 HAM10000 classes")

        model_status = "Available (Loaded)" if os.path.exists(MODEL_PATH) else "Not Found (Training Required)"
        status_color = "green" if os.path.exists(MODEL_PATH) else "red"
        st.markdown(f"**Model Status**: :{status_color}[{model_status}]")

    # Main Tabs
    tab_inspect, tab_eval, tab_dataset = st.tabs([
        "🔬 Dermoscopic Analysis & Grad-CAM",
        "📊 Model Evaluation & Metrics",
        "📚 Dataset & Clinical Reference"
    ])

    with tab_inspect:
        col_input, col_results = st.columns([1, 1.25], gap="large")

        with col_input:
            st.markdown("### 1. Image Input")
            
            # Check for sample images
            sample_files = glob.glob(os.path.join(SAMPLE_DIR, "*.*"))
            if not sample_files:
                sample_files = create_demo_samples(SAMPLE_DIR)

            sample_options = ["None (Upload my own image)"] + [os.path.basename(p) for p in sorted(sample_files)]
            # Default to index 0 (None) to ensure clean empty state on startup
            selected_sample = st.selectbox("Select sample test lesion:", sample_options, index=0)

            uploaded_file = st.file_uploader(
                "Or upload dermoscopic photograph (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="Upload a clinical dermoscopic photograph for real-time inference."
            )

            active_image = None
            image_source_label = ""
            image_bytes = None
            image_hash = ""

            # Explicit stream handling: uploaded file takes precedence if provided
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    if len(image_bytes) == 0:
                        st.error("Uploaded file is empty (0 bytes). Please upload a valid image.")
                        active_image = None
                    else:
                        image_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
                        # Decode image bytes freshly into RGB PIL Image
                        active_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        image_source_label = f"Uploaded: {uploaded_file.name} ({active_image.width}×{active_image.height} px)"
                except Exception as e:
                    st.error(f"Failed to decode uploaded image: {e}")
                    active_image = None
            elif selected_sample != "None (Upload my own image)":
                chosen_path = os.path.join(SAMPLE_DIR, selected_sample)
                if os.path.exists(chosen_path):
                    try:
                        with open(chosen_path, "rb") as f:
                            image_bytes = f.read()
                        image_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
                        active_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        image_source_label = f"Sample: {selected_sample} ({active_image.width}×{active_image.height} px)"
                    except Exception as e:
                        st.error(f"Failed to read sample image: {e}")
                        active_image = None

            if active_image is not None:
                st.image(active_image, caption=image_source_label, use_container_width=True)
                if uploaded_file is not None and selected_sample != "None (Upload my own image)":
                    st.caption("ℹ️ *Notice: Uploaded file takes precedence over sample dropdown selection.*")
            else:
                st.info("Upload an image or select a sample to begin.")

        with col_results:
            st.markdown("### 2. Neural Network Prediction & Explainability")

            if active_image is None:
                st.markdown("""
                <div style="background:#161b22;border:1px dashed #30363d;border-radius:10px;padding:48px 24px;text-align:center;color:#8b949e;">
                    <div style="font-size:2.2rem;margin-bottom:12px;">🔬</div>
                    <div style="font-size:1.15rem;font-weight:600;color:#e6edf3;margin-bottom:8px;">
                        Upload an image or select a sample to begin.
                    </div>
                    <div style="font-size:0.9rem;color:#8b949e;line-height:1.5;">
                        Select a sample lesion from the dropdown or upload your own dermoscopic photo to view AI classification and Grad-CAM explainability.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                model = get_model()
                if model is None:
                    st.error(
                        "**Model Artifact Missing**: `models/skin_lesion_model.keras` has not been generated yet.\n\n"
                        "Run the training script in your terminal:\n"
                        "```bash\npython train_model.py\n```"
                    )
                else:
                    try:
                        # Preprocess image freshly for MobileNetV2
                        pil_rgb, tensor = preprocess_pil(active_image, target_size=(224, 224))
                        
                        # Run fresh inference on the tensor
                        probs = model.predict(tensor, verbose=0)[0]
                        pred_idx = int(np.argmax(probs))
                        pred_code = classes[pred_idx]
                        confidence = float(probs[pred_idx])

                        meta = CLASS_METADATA.get(pred_code, {
                            "name": pred_code,
                            "category": "Classification Category",
                            "risk_level": "Undetermined",
                            "description": "Standard HAM10000 category"
                        })

                        # Determine alert color by risk
                        is_malignant = "malignant" in meta["category"].lower() or "critical" in meta["risk_level"].lower()
                        metric_val_class = "metric-value-high" if is_malignant else "metric-value-benign"

                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Predicted Diagnosis ({pred_code})</div>
                            <div class="{metric_val_class}">{meta['name']}</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 4px;">
                                <strong>Category:</strong> {meta['category']} &nbsp;|&nbsp; 
                                <strong>Risk Level:</strong> {meta['risk_level']}
                            </div>
                            <div style="margin-top: 10px; color: #8b949e; font-size: 0.88rem;">
                                {meta['description']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Model Confidence Score", f"{confidence * 100:.2f}%")
                        with col_m2:
                            st.metric("Risk Classification", meta["category"])

                        # Grad-CAM Section
                        st.markdown("#### Visual Explanation (Grad-CAM)")
                        from gradcam import compute_gradcam, overlay_gradcam

                        heatmap = compute_gradcam(model, tensor, class_idx=pred_idx)
                        colored_map, blended = overlay_gradcam(
                            pil_rgb, heatmap, alpha=overlay_alpha, colormap_name=colormap
                        )

                        cam_c1, cam_c2, cam_c3 = st.columns(3)
                        with cam_c1:
                            st.image(pil_rgb, caption="Input (224×224)", use_container_width=True)
                        with cam_c2:
                            st.image(colored_map, caption=f"Grad-CAM ({colormap})", use_container_width=True)
                        with cam_c3:
                            st.image(blended, caption="Explainability Overlay", use_container_width=True)

                        st.markdown("""
                        <div class="callout-explain">
                            💡 <strong>Interpretability Notice:</strong> The warm (red/yellow) areas indicate regions 
                            in the dermoscopic image that contributed most strongly to the convolutional neural network's 
                            decision for the predicted class. This highlights visual attention patterns, but does not 
                            constitute an anatomically or histopathologically validated medical boundary.
                        </div>
                        """, unsafe_allow_html=True)

                        # Diagnostic Information Panel (Live recalculation verification)
                        with st.expander("🛠️ Live Diagnostic Telemetry (Recalculation Verification)", expanded=True):
                            diag_c1, diag_c2 = st.columns(2)
                            with diag_c1:
                                st.markdown(f"**Selected Source:** `{image_source_label}`")
                                st.markdown(f"**Image Dimensions:** `{active_image.width} × {active_image.height}` px")
                                st.markdown(f"**Input Tensor Shape:** `{tensor.shape}`")
                                st.markdown(f"**Image SHA-256 Digest:** `{image_hash}`")
                            with diag_c2:
                                st.markdown(f"**Inference Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
                                st.markdown(f"**Top Predicted Class:** `{pred_code}` ({meta['name']})")
                                st.markdown(f"**Top Confidence Score:** `{confidence * 100:.2f}%`")
                                st.markdown("**Inference Mode:** Freshly computed (un-cached)")
                            
                            st.markdown("**Live Prediction Probability Vector across all 7 HAM10000 Categories:**")
                            prob_summary = {cls: f"{p:.5f} ({p*100:.2f}%)" for cls, p in zip(classes, probs)}
                            st.json(prob_summary)

                        # Probability Distribution Bar Chart
                        with st.expander("📊 View All 7 Diagnostic Category Probabilities", expanded=False):
                            prob_dict = {
                                f"{c} ({CLASS_METADATA.get(c, {}).get('name', c)[:18]})": float(probs[i])
                                for i, c in enumerate(classes)
                            }
                            df_probs = pd.DataFrame({
                                "Category Code": classes,
                                "Full Name": [CLASS_METADATA.get(c, {}).get("name", c) for c in classes],
                                "Confidence": [f"{p * 100:.2f}%" for p in probs],
                                "Raw Score": probs
                            }).sort_values(by="Raw Score", ascending=False)

                            st.dataframe(
                                df_probs[["Category Code", "Full Name", "Confidence"]],
                                use_container_width=True,
                                hide_index=True
                            )
                            st.bar_chart(prob_dict)

                    except Exception as e:
                        st.error(f"Inference or Grad-CAM computation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")

        st.markdown("""
        <div class="callout-warning">
            ⚠️ <strong>Academic &amp; Clinical Safety Disclaimer:</strong> This artificial intelligence model is developed exclusively 
            for university research and computer vision educational demonstrations. It is <strong>NOT</strong> an FDA-approved or CE-marked medical 
            diagnostic tool and must never be utilized to diagnose, monitor, rule out, or guide medical therapy for skin malignancies. 
            Always consult a board-certified dermatologist for clinical skin examinations.
        </div>
        """, unsafe_allow_html=True)

    with tab_eval:
        st.markdown("### Model Validation & Test Performance")
        st.caption("Evaluated on held-out test split partitioned without lesion-group leakage.")

        if not os.path.exists(EVAL_PATH):
            st.info(
                "**Evaluation Results Pending**: Test set evaluation has not been executed yet.\n\n"
                "Run the evaluation script to compute metrics and confusion matrix on the held-out test set:\n"
                "```bash\npython evaluate_model.py\n```"
            )
        else:
            try:
                with open(EVAL_PATH, "r") as f:
                    eval_data = json.load(f)

                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                with col_e1:
                    st.metric("Test Accuracy", f"{eval_data.get('accuracy', 0.0) * 100:.2f}%")
                with col_e2:
                    macro_f1 = eval_data.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0)
                    st.metric("Macro F1-Score", f"{macro_f1:.4f}")
                with col_e3:
                    weighted_f1 = eval_data.get("classification_report", {}).get("weighted avg", {}).get("f1-score", 0.0)
                    st.metric("Weighted F1-Score", f"{weighted_f1:.4f}")
                with col_e4:
                    st.metric("Test Samples Evaluated", eval_data.get("sample_count", "N/A"))

                st.divider()

                col_cm, col_report = st.columns([1.1, 1], gap="medium")
                with col_cm:
                    st.markdown("#### Test Set Confusion Matrix")
                    if os.path.exists(CM_PATH):
                        st.image(CM_PATH, use_container_width=True)
                    else:
                        st.info("Confusion matrix image not found.")

                with col_report:
                    st.markdown("#### Per-Class Metrics")
                    report_dict = eval_data.get("classification_report", {})
                    table_rows = []
                    for c in classes:
                        if c in report_dict:
                            metrics = report_dict[c]
                            table_rows.append({
                                "Class": c,
                                "Precision": f"{metrics.get('precision', 0):.3f}",
                                "Recall": f"{metrics.get('recall', 0):.3f}",
                                "F1-Score": f"{metrics.get('f1-score', 0):.3f}",
                                "Test Count": metrics.get("support", 0)
                            })
                    if table_rows:
                        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error parsing evaluation report: {e}")

    with tab_dataset:
        st.markdown("### The HAM10000 Dataset Reference")
        st.markdown("""
        The **HAM10000** ("Human Against Machine with 10000 training images") dataset is a benchmark 
        dermatoscopic image collection collected by the Medical University of Vienna and the Cliff Rosendahl 
        dermatology clinic in Queensland, Australia.

        #### Canonical 7 Diagnostic Categories
        """)

        meta_rows = []
        for code, info in CLASS_METADATA.items():
            meta_rows.append({
                "Dataset Code": code,
                "Formal Medical Name": info["name"],
                "Category": info["category"],
                "Clinical Risk": info["risk_level"],
                "Description": info["description"]
            })
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

        st.markdown("""
        #### Technical Methodology Highlights
        - **Leakage Prevention**: HAM10000 contains multiple lesion photos of the same patient taken from different angles or times. Group-aware splitting (`GroupShuffleSplit` on `lesion_id`) prevents images of the same lesion from appearing in both training and test sets.
        - **Transfer Learning**: Pretrained MobileNetV2 features from ImageNet are reused to extract rich spatial textures, followed by dropout regularized dense classification.
        - **Grad-CAM Architecture**: Gradients of the predicted class with respect to the `out_relu` activation map are pooled and weighted to generate high-resolution visual explanations without retraining.
        """)

    # Footer
    st.markdown("""
    <hr style="border-color:#30363d;margin-top:40px;">
    <div style="text-align:center;color:#8b949e;font-size:0.8rem;padding-bottom:20px;">
        Explainable Skin Cancer Detection System • MobileNetV2 + Grad-CAM • Academic Prototype • Strictly Non-Clinical Use
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
