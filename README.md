# ❤️ CARDIOVISION

**AI-Based Cardiovascular Disease Detection & Explainable ECG Analysis**

An end-to-end system that reads raw 12-lead ECG recordings from the PTB-XL dataset, preprocesses signals, converts them into CWT scalograms, trains a Vision Transformer from scratch for cardiovascular disease classification, and provides Grad-CAM explanations with a user-friendly Streamlit interface.

---

## 🏗️ Architecture

```
PTB-XL WFDB ECG → Signal Quality Check → Butterworth Filter → Pan-Tompkins R-Peak Detection
→ Min-Max Normalization → CWT Scalogram → Vision Transformer (from scratch) → 5-Class Prediction
→ Grad-CAM Explanation → Streamlit Dashboard
```

## 🎯 Target Classes

| # | Class | Description |
|---|-------|-------------|
| 0 | Normal | Normal ECG |
| 1 | Myocardial Infarction | MI findings |
| 2 | Arrhythmia | Conduction disturbances & rhythm abnormalities |
| 3 | Left Ventricular Hypertrophy | Hypertrophy findings |
| 4 | ST/T Wave Abnormalities | ST-segment or T-wave changes |

## 📁 Project Structure

```
cardiovision/
├── app.py                    # Streamlit application
├── config.yaml               # Central configuration
├── requirements.txt          # Python dependencies
├── README.md
├── src/
│   ├── data/                 # Dataset loading & label mapping
│   ├── preprocessing/        # Signal processing pipeline
│   ├── models/               # Custom Vision Transformer
│   ├── training/             # Training & evaluation
│   ├── explainability/       # Grad-CAM for ViT
│   ├── inference/            # Prediction pipeline
│   ├── visualization/        # Plotting utilities
│   └── utils/                # Config, seed, logging
├── scripts/                  # Execution scripts
├── tests/                    # Unit tests
├── data/                     # Processed data & splits
├── models/                   # Checkpoints
└── outputs/                  # Figures, metrics, explanations
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python scripts/prepare_dataset.py
```

### 3. Generate CWT Scalograms
```bash
python scripts/generate_cwt.py
```

### 4. Train Model
```bash
python scripts/train_model.py
```

### 5. Evaluate Model
```bash
python scripts/evaluate_model.py
```

### 6. Launch Application
```bash
streamlit run app.py
```

## ⚕️ Disclaimer

> **This is an AI-assisted analysis tool for research purposes only.** It does not provide a definitive medical diagnosis. Always consult a qualified healthcare professional for clinical decisions.

## 📊 Dataset

Uses the [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) dataset:
- 21,800 twelve-lead ECG records
- 10-second recordings at 100 Hz and 500 Hz
- 71 SCP diagnostic codes mapped to 5 target classes

## 🔧 Configuration

All hyperparameters are configurable via `config.yaml`. Key settings:

| Parameter | Default |
|-----------|---------|
| Sampling Rate | 100 Hz |
| Butterworth Band | 0.5–40 Hz |
| CWT Wavelet | Morlet |
| Image Size | 224×224 |
| ViT Layers | 6 |
| ViT Heads | 8 |
| Embedding Dim | 256 |
| Batch Size | 16 |
| Learning Rate | 1e-4 |
| Epochs | 50 |

## 🧪 Testing

```bash
python -m pytest tests/ -v
```
