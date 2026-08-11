"""
CARDIOVISION - Streamlit App
Interactive Web UI for Cardiovascular Disease Detection and Explainability.
"""

import os
import sys
from pathlib import Path
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.models.vit import CardioViT
from src.inference.predict import CardiovisionPredictor
from src.visualization.ecg_plot import plot_12_lead_ecg, plot_signal_with_r_peaks
from src.visualization.cwt_plot import plot_scalogram, plot_overlay
from src.visualization.results_plot import plot_training_history, plot_confusion_matrix

# ==============================================================================
# Configuration & Setup
# ==============================================================================

cfg = load_config()

st.set_page_config(
    page_title=cfg.streamlit.page_title,
    page_icon=cfg.streamlit.page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium styling
st.markdown("""
<style>
    :root {
        --primary-color: #ff4b4b;
        --bg-color: #0e1117;
        --secondary-bg: #262730;
        --text-color: #fafafa;
        --text-muted: #9e9e9e;
    }
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #ff8f00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: var(--text-muted);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: var(--secondary-bg);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(255,75,75,0.2);
    }
    .prob-bar {
        height: 10px;
        border-radius: 5px;
        background-color: #333;
        margin-top: 5px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff8f00);
        transition: width 0.5s ease-out;
    }
    .disclaimer-box {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load model and build predictor (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CardioViT(
        image_size=cfg.model.input_size[0],
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.input_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        use_cls_token=cfg.model.use_cls_token,
        use_positional_embedding=cfg.model.use_positional_embedding
    )
    
    checkpoint_path = Path(cfg.output.checkpoints_dir) / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"Model loaded (Epoch {checkpoint['epoch']})")
    else:
        st.sidebar.warning("No trained checkpoint found! Using random initialization.")
        
    return CardiovisionPredictor(model, cfg, device)


predictor = load_predictor()

# ==============================================================================
# UI Components
# ==============================================================================

def render_sidebar():
    st.sidebar.markdown("## Navigation")
    pages = [
        "Home",
        "ECG Analysis (Upload)",
        "Model Performance"
    ]
    selection = st.sidebar.radio("Go to", pages)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    
    # Allow user to tweak basic inference settings
    st.session_state.show_heatmap = st.sidebar.checkbox("Show Grad-CAM", value=True)
    
    return selection

def display_disclaimer():
    st.markdown(f'<div class="disclaimer-box">{cfg.streamlit.disclaimer}</div>', unsafe_allow_html=True)

# ==============================================================================
# Pages
# ==============================================================================

def page_home():
    st.markdown('<div class="main-header">CardioVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Based Cardiovascular Disease Detection & Explainable ECG Analysis</div>', unsafe_allow_html=True)
    
    display_disclaimer()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 About")
        st.markdown("""
        CardioVision is an end-to-end system that analyzes raw 12-lead ECG recordings to detect 
        cardiovascular diseases. It uses advanced signal processing and a custom Vision Transformer (ViT) 
        trained from scratch.
        """)
        
        st.markdown("### 🧬 Target Classes")
        for cls in cfg.labels.class_names:
            st.markdown(f"- **{cls}**")
            
    with col2:
        st.markdown("### ⚙️ Pipeline")
        st.markdown("""
        1. **Signal Loading**: Reads raw WFDB ECG files (.dat/.hea)
        2. **Preprocessing**: Quality check, Butterworth bandpass filter
        3. **Feature Extraction**: R-Peak detection (Pan-Tompkins)
        4. **Transformation**: Continuous Wavelet Transform (CWT) scalograms
        5. **Prediction**: Custom Vision Transformer (CardioViT)
        6. **Explainability**: ViT-compatible Grad-CAM
        """)

def page_analysis():
    st.markdown("## ECG Analysis")
    display_disclaimer()
    
    st.markdown("### Upload ECG Record")
    st.info("Please upload both `.dat` and `.hea` files for a single ECG record.")
    
    uploaded_files = st.file_uploader("Upload WFDB files", type=['dat', 'hea'], accept_multiple_files=True)
    
    if uploaded_files:
        dat_file = next((f for f in uploaded_files if f.name.endswith('.dat')), None)
        hea_file = next((f for f in uploaded_files if f.name.endswith('.hea')), None)
        
        if dat_file and hea_file:
            with st.spinner("Processing ECG..."):
                # Save to temp dir for WFDB to read
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Strip extensions to match WFDB requirements
                    record_name = dat_file.name[:-4]
                    
                    dat_path = Path(tmpdirname) / dat_file.name
                    hea_path = Path(tmpdirname) / hea_file.name
                    
                    with open(dat_path, "wb") as f:
                        f.write(dat_file.getbuffer())
                    with open(hea_path, "wb") as f:
                        f.write(hea_file.getbuffer())
                        
                    # Run Inference
                    record_path = str(Path(tmpdirname) / record_name)
                    results = predictor.predict_from_file(record_path, generate_explanation=st.session_state.show_heatmap)
                    
            if results["status"] == "success":
                st.success("Analysis complete!")
                render_analysis_results(results)
            else:
                st.error(f"Error during analysis: {results['message']}")
        else:
            st.warning("Please upload BOTH the .dat and .hea files.")

def render_analysis_results(results):
    st.markdown("---")
    
    # 1. Prediction Results
    st.markdown("### 🩺 Prediction Results")
    
    # Main Prediction Metric Card
    pred_class = results["predicted_class_name"]
    confidence = results["confidence"] * 100
    
    color = "#4CAF50" if pred_class == "Normal" else "#ff4b4b"
    
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <h3 style="margin:0; color: var(--text-muted);">Primary Finding</h3>
        <h1 style="margin: 10px 0; color: {color};">{pred_class}</h1>
        <h4 style="margin:0; font-weight: normal;">Confidence: <strong>{confidence:.1f}%</strong></h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Probabilities breakdown
    st.markdown("#### Probability Distribution")
    for cls_name, prob in sorted(results["probabilities"].items(), key=lambda x: x[1], reverse=True):
        p_pct = prob * 100
        bar_color = "#ff4b4b" if cls_name == pred_class else "#666"
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span>{cls_name}</span>
                <span>{p_pct:.1f}%</span>
            </div>
            <div class="prob-bar">
                <div class="prob-fill" style="width: {p_pct}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. Explainability
    if "gradcam_overlay" in results:
        st.markdown("### 🔍 Model Explainability (Grad-CAM)")
        st.markdown("Highlights regions of the CWT scalogram that most strongly influenced the prediction.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Scalogram**")
            # transpose CHW to HWC for matplotlib/PIL
            scalo_hwc = results["scalogram_chw"].transpose(1, 2, 0)
            fig_scalo = plot_scalogram(scalo_hwc, title="")
            st.pyplot(fig_scalo)
            
        with col2:
            st.markdown("**Attention Overlay**")
            fig_overlay = plot_overlay(results["gradcam_overlay"], title="")
            st.pyplot(fig_overlay)
            
        st.markdown("---")

    # 3. Preprocessing Visualization
    st.markdown("### 📈 Signal Processing Pipeline")
    
    tab1, tab2, tab3 = st.tabs(["Raw Signal", "Filtered & R-Peaks", "Heart Rate Analysis"])
    
    with tab1:
        st.markdown("Standard 12-Lead layout of the raw uploaded signal.")
        fig_raw = plot_12_lead_ecg(results["raw_signal"], cfg.dataset.sampling_rate, title="Raw 12-Lead ECG")
        st.pyplot(fig_raw)
        
    with tab2:
        st.markdown("Butterworth filtered signal with Pan-Tompkins R-peak detection on Lead II.")
        fig_filtered = plot_signal_with_r_peaks(
            results["filtered_signal"], 
            results["r_peaks"], 
            cfg.dataset.sampling_rate,
            lead_idx=cfg.preprocessing.pan_tompkins.primary_lead
        )
        st.pyplot(fig_filtered)
        
    with tab3:
        rr_intervals = results["rr_intervals"]
        if len(rr_intervals) > 0:
            hr = 60 * cfg.dataset.sampling_rate / np.mean(rr_intervals)
            st.metric("Estimated Heart Rate", f"{hr:.0f} BPM")
            
            # Simple RR interval plot
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(rr_intervals / cfg.dataset.sampling_rate, marker='o')
            ax.set_title("RR Intervals over Time")
            ax.set_xlabel("Beat index")
            ax.set_ylabel("RR Interval (s)")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.warning("Insufficient R-peaks detected for heart rate analysis.")

def page_performance():
    st.markdown("## Model Performance")
    
    import json
    
    # Load test metrics
    metrics_path = Path(cfg.output.metrics_dir) / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.2%}")
        col3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.2%}")
        
        st.markdown("### Confusion Matrix")
        if "confusion_matrix" in metrics:
            fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], cfg.labels.class_names)
            st.pyplot(fig_cm)
            
        st.markdown("### Per-Class Metrics")
        if "per_class" in metrics:
            df_metrics = pd.DataFrame(metrics["per_class"]).T
            df_metrics = df_metrics.applymap(lambda x: f"{x:.2%}")
            st.dataframe(df_metrics, use_container_width=True)
            
    else:
        st.info("Test metrics not found. Run `scripts/evaluate_model.py` first.")
        
    st.markdown("---")
    st.markdown("### Training History")
    log_path = Path(cfg.output.training_log)
    if log_path.exists():
        fig_hist = plot_training_history(str(log_path))
        st.pyplot(fig_hist)
    else:
        st.info("Training history not found. Run `scripts/train_model.py` first.")

# ==============================================================================
# Main App Loop
# ==============================================================================

def main():
    selection = render_sidebar()
    
    if selection == "Home":
        page_home()
    elif selection == "ECG Analysis (Upload)":
        page_analysis()
    elif selection == "Model Performance":
        page_performance()

if __name__ == "__main__":
    main()
