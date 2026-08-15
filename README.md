# ❤️ CARDIOVISION

**AI-Based Cardiovascular Disease Detection & Explainable ECG Analysis**

An end-to-end system that reads raw 12-lead ECG recordings from the PTB-XL dataset, preprocesses signals, converts them into CWT scalograms, trains a Vision Transformer from scratch for cardiovascular disease classification, and provides Grad-CAM explanations with both a Streamlit interface and a modern Next.js web application.

---

## 📖 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [System Architecture](#-system-architecture)
- [Target Classes](#-target-classes)
- [Project Structure](#-project-structure)
- [Complete Setup Guide](#-complete-setup-guide)
- [Training the Model](#-training-the-model)
- [Running the Applications](#-running-the-applications)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 What This Project Does

CardioVision is an AI system that helps detect cardiovascular diseases from ECG (electrocardiogram) readings. Think of it as a smart assistant that:

1. **Reads ECG files** - Takes raw heart signal recordings (the squiggly lines doctors look at)
2. **Cleans the data** - Removes noise and filters out unwanted signals
3. **Finds patterns** - Uses AI to identify signs of heart problems
4. **Explains its reasoning** - Shows which parts of the ECG influenced the diagnosis
5. **Presents results** - Displays predictions in easy-to-understand web interfaces

**Important:** This is a research tool, NOT a medical device. Always consult a real doctor for health concerns!

---

## 🏗️ System Architecture

```
Raw ECG File (.dat/.hea)
         ↓
   Signal Quality Check
         ↓
   Butterworth Filter (removes noise)
         ↓
   R-Peak Detection (finds heartbeats)
         ↓
   Normalization (standardizes values)
         ↓
   CWT Transformation (converts to image)
         ↓
   Vision Transformer AI Model
         ↓
   Prediction + Grad-CAM Explanation
         ↓
   Web Interface (Frontend + Backend)
```

### Technology Stack

**Machine Learning:**
- **PyTorch** - Deep learning framework
- **Custom Vision Transformer** - AI model built from scratch (not pre-trained)
- **Grad-CAM** - Explainability technique to visualize what the AI "sees"

**Signal Processing:**
- **WFDB** - Reads ECG files from PhysioNet format
- **SciPy** - Signal filtering (Butterworth bandpass)
- **PyWavelets** - Continuous Wavelet Transform (CWT)

**Backend:**
- **FastAPI** - Modern Python API server
- **Uvicorn** - ASGI server for FastAPI

**Frontend:**
- **Next.js 16** - React framework for web interface
- **TypeScript** - Type-safe JavaScript
- **TailwindCSS** - Styling framework
- **Shadcn/ui** - UI component library

**Additional Tools:**
- **Streamlit** - Alternative web interface for quick demos
- **Matplotlib/Seaborn** - Data visualization

---

## 🎯 Target Classes

The model classifies ECGs into 5 categories:

| Class | Description | What it means |
|-------|-------------|---------------|
| **Normal** | Normal ECG | Healthy heart rhythm |
| **Myocardial Infarction** | Heart attack findings | Damaged heart muscle from blocked blood flow |
| **Arrhythmia** | Irregular heartbeat | Abnormal heart rhythm patterns |
| **Left Ventricular Hypertrophy** | Enlarged left heart chamber | Heart muscle thickening (often from high blood pressure) |
| **ST/T Wave Abnormalities** | Abnormal ECG wave shapes | Possible heart problems affecting electrical signals |

---

## 📁 Project Structure

```
cardiovision/
├── 📄 app.py                          # Streamlit web application
├── ⚙️ config.yaml                     # All settings and hyperparameters
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # This file
│
├── 🧠 src/                            # Source code (Python modules)
│   ├── data/                          # Dataset loading & label mapping
│   ├── preprocessing/                 # Signal processing (filtering, CWT, etc.)
│   ├── models/                        # Vision Transformer model definition
│   ├── training/                      # Training & evaluation logic
│   ├── explainability/                # Grad-CAM visualization
│   ├── inference/                     # Prediction pipeline
│   ├── visualization/                 # Plotting utilities
│   └── utils/                         # Helper functions (config, logging, etc.)
│
├── 🎬 scripts/                        # Executable scripts
│   ├── prepare_dataset.py             # Step 1: Parse data and create splits
│   ├── generate_cwt.py                # Step 2: Generate scalogram images
│   ├── train_model.py                 # Step 3: Train the AI model
│   └── evaluate_model.py              # Step 4: Test model performance
│
├── 🔙 backend/                        # FastAPI server
│   ├── main.py                        # API endpoints
│   ├── __init__.py
│   └── README.md
│
├── 🖥️ frontend/                       # Next.js web application
│   ├── app/                           # Next.js pages
│   ├── components/                    # React components
│   ├── public/                        # Static assets
│   ├── package.json                   # Node.js dependencies
│   └── README.md
│
├── 📊 dataset/                        # PTB-XL dataset (downloaded separately)
│   ├── ptbxl_database.csv             # Metadata
│   ├── records100/                    # ECG files at 100 Hz
│   └── records500/                    # ECG files at 500 Hz
│
├── 💾 data/                           # Processed data
│   ├── processed/cwt_images/          # Generated scalogram images
│   └── splits/                        # Train/val/test splits
│
├── 🏋️ models/                         # Saved model checkpoints
│   └── checkpoints/best_model.pth     # Trained model weights
│
├── 📈 outputs/                        # Training outputs
│   ├── figures/                       # Plots and visualizations
│   ├── metrics/                       # Performance metrics
│   ├── predictions/                   # Model predictions
│   └── training_log.csv               # Training history
│
└── 🧪 tests/                          # Unit tests
```

---

## 🚀 Complete Setup Guide

Follow these steps carefully if you're new to Python projects.

### Prerequisites

Before starting, make sure you have:
- **Python 3.9 or higher** installed ([Download Python](https://www.python.org/downloads/))
- **Node.js 18+** and npm installed ([Download Node.js](https://nodejs.org/))
- **Git** (optional, for cloning the repository)
- At least **20 GB free disk space** (for dataset and generated images)
- **Windows, macOS, or Linux** operating system

To check if Python and Node.js are installed:
```bash
python --version    # Should show 3.9 or higher
node --version      # Should show v18 or higher
npm --version       # Should show 9 or higher
```

---

### Step 1: Download the Project

If you have Git:
```bash
git clone <repository-url>
cd cardiovision
```

Or download the ZIP file and extract it, then open a terminal in that folder.

---

### Step 2: Set Up Python Virtual Environment

A virtual environment keeps this project's dependencies separate from other Python projects.

**On Windows:**
```cmd
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

You'll know it's activated when you see `(.venv)` at the start of your terminal prompt.

**To deactivate later:**
```bash
deactivate
```

---

### Step 3: Install Python Dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning)
- FastAPI (backend server)
- Streamlit (UI)
- WFDB (ECG file reader)
- NumPy, Pandas, SciPy (data processing)
- Matplotlib (visualization)
- And many other libraries...

**This may take 5-10 minutes** depending on your internet speed.

---

### Step 4: Download the PTB-XL Dataset

The model needs ECG data to train on. Download the PTB-XL dataset:

1. Visit: https://physionet.org/content/ptb-xl/1.0.3/
2. Click **"Download the ZIP file"** (about 2.5 GB)
3. Extract the contents
4. Copy/move the extracted files into the `dataset/` folder in your project

Your `dataset/` folder should look like:
```
dataset/
├── ptbxl_database.csv
├── scp_statements.csv
├── records100/
│   ├── 00000/
│   ├── 00001/
│   └── ...
└── records500/
    └── ...
```

---

### Step 5: Set Up the Frontend (Next.js)

Open a **new terminal window** and navigate to the frontend folder:

```bash
cd frontend
npm install
```

This installs all Node.js packages needed for the web interface.

**Create environment file** (optional, for custom API URL):
```bash
# In frontend/ folder
echo NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 > .env.local
```

---

### Step 6: Install Backend Dependencies

The backend dependencies are already in `requirements.txt`, but you can verify:

```bash
# Make sure virtual environment is activated
pip install fastapi "uvicorn[standard]" python-multipart
```

---

## 🏋️ Training the Model

Now that everything is set up, follow these steps to train your own AI model.

### Step 1: Prepare the Dataset (10-15 minutes)

This script reads the ECG metadata, maps diagnostic codes to our 5 classes, and splits the data into training/validation/test sets.

```bash
# Make sure .venv is activated
python scripts/prepare_dataset.py
```

**What this does:**
- Loads `ptbxl_database.csv` (patient metadata)
- Maps complex medical codes to 5 simple classes
- Splits data: 70% training, 15% validation, 15% testing
- Saves splits to `data/splits/`

**Output files:**
- `data/splits/train_records.csv`
- `data/splits/val_records.csv`
- `data/splits/test_records.csv`
- `data/splits/split_statistics.json`

---

### Step 2: Generate CWT Scalograms (1-2 hours)

This converts raw ECG signals into images that the AI can process.

```bash
python scripts/generate_cwt.py
```

**What this does:**
- Reads each ECG file (`.dat` and `.hea`)
- Applies signal quality checks
- Filters noise using Butterworth filter
- Detects R-peaks (heartbeats)
- Applies Continuous Wavelet Transform
- Saves as 224×224 PNG images

**Output:**
- ~21,800 images in `data/processed/cwt_images/`
- `preprocessing_failed_ecgs.csv` (records that failed)

**Progress bar** will show you how many files are processed.

**Why this takes time:** Processing 21,800 ECG files requires significant computation.

---

### Step 3: Train the Model (2-4 hours with GPU, 10+ hours with CPU)

This is where the AI learns to recognize heart conditions.

```bash
python scripts/train_model.py
```

**What this does:**
- Loads CWT images and labels
- Creates the Vision Transformer model (CardioViT)
- Trains for 40 epochs (passes through all data)
- Validates after each epoch
- Saves the best model to `models/checkpoints/best_model.pth`
- Logs training progress to `outputs/training_log.csv`

**Monitor progress:**
- Training/validation loss (should decrease)
- Accuracy and F1 scores (should increase)
- Best model is saved automatically

**GPU vs CPU:**
- With NVIDIA GPU: ~2-4 hours
- With CPU only: 10+ hours (be patient!)

To check if PyTorch sees your GPU:
```python
python -c "import torch; print(torch.cuda.is_available())"
# True = GPU available, False = CPU only
```

---

### Step 4: Evaluate the Model (5-10 minutes)

Test the trained model on unseen data:

```bash
python scripts/evaluate_model.py
```

**What this does:**
- Loads the best trained model
- Runs predictions on test set (15% of data)
- Calculates metrics: accuracy, F1 score, confusion matrix
- Saves results to `outputs/metrics/test_metrics.json`

**Expected Performance:**
- Accuracy: ~55-60%
- Macro F1: ~45-50%

(These numbers vary based on training run and are for a challenging 5-class medical problem!)

---

## 🖥️ Running the Applications

You have three ways to interact with the trained model:

### Option 1: Streamlit App (Simplest)

Great for quick testing and demos.

```bash
# Make sure .venv is activated
streamlit run app.py
```

Opens in browser at `http://localhost:8501`

**Features:**
- Upload ECG files (.dat + .hea)
- View predictions with confidence scores
- See Grad-CAM explanations
- Explore model performance metrics

---

### Option 2: Full Stack Web App (Backend + Frontend)

Professional web interface with modern UI.

**Terminal 1 - Start Backend:**
```bash
# Make sure .venv is activated
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend runs at `http://127.0.0.1:8000`

**API Endpoints:**
- `POST /api/analyze` - Upload and analyze ECG
- `GET /api/sample` - Analyze random sample
- `GET /api/model-info` - Get model details

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```

Frontend runs at `http://localhost:3000`

**What you get:**
- Modern, responsive web interface
- Drag-and-drop ECG upload
- Real-time processing with progress indicators
- Beautiful visualizations
- Risk level indicators
- Heart rate calculation

---

### Option 3: Python API (For Developers)

Use the model programmatically in your own Python scripts:

```python
from pathlib import Path
import torch
from src.models.vit import CardioViT
from src.inference.predict import CardiovisionPredictor
from src.utils.config import load_config

# Load configuration
cfg = load_config()

# Create model
model = CardioViT(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=5,
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    mlp_dim=512
)

# Load trained weights
checkpoint = torch.load("models/checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Create predictor
predictor = CardiovisionPredictor(model, cfg, device="cuda")

# Make prediction
result = predictor.predict_from_file("path/to/ecg_record")

print(f"Predicted class: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## ⚙️ Configuration

All settings are in `config.yaml`. Here are the key parameters:

### Dataset Settings
```yaml
dataset:
  sampling_rate: 100          # Use 100 Hz or 500 Hz
  root_dir: "./dataset"       # Where ECG files are stored
```

### Preprocessing Settings
```yaml
preprocessing:
  butterworth:
    low_cutoff_hz: 0.5        # High-pass filter (removes baseline drift)
    high_cutoff_hz: 40.0      # Low-pass filter (removes high-freq noise)
  
  pan_tompkins:
    primary_lead: 1           # Lead II used for R-peak detection
```

### Model Architecture
```yaml
model:
  name: "CardioViT"
  patch_size: 16              # Divide 224×224 image into 16×16 patches
  embedding_dim: 256          # Size of patch embeddings
  num_layers: 6               # Number of transformer blocks
  num_heads: 8                # Attention heads per layer
  mlp_dim: 512                # Hidden layer size
  dropout: 0.1                # Dropout rate (prevents overfitting)
```

### Training Hyperparameters
```yaml
training:
  epochs: 40                  # How many times to see all data
  batch_size: 16              # Images per training step
  
  optimizer:
    learning_rate: 0.0001     # Step size for weight updates
    weight_decay: 0.01        # L2 regularization
  
  loss:
    use_class_weights: true   # Handle imbalanced classes
    label_smoothing: 0.1      # Soft labels (improves generalization)
  
  mixed_precision: true       # Faster training on modern GPUs
```

**To modify settings:**
1. Open `config.yaml` in a text editor
2. Change values
3. Save file
4. Re-run training script

---

## 🧪 Testing

Run automated tests to verify everything works:

```bash
# Make sure .venv is activated
python -m pytest tests/ -v
```

**What this tests:**
- Signal preprocessing functions
- CWT transformation
- Model forward pass
- Data loading pipeline

All tests should pass ✅

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Solution:** Activate virtual environment first:
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

---

### "CUDA out of memory"

Your GPU doesn't have enough memory.

**Solution 1:** Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 8  # or even 4
```

**Solution 2:** Use CPU (slower but works):
```yaml
device:
  use_cuda: false
```

---

### "Cannot find checkpoint file"

You haven't trained a model yet.

**Solution:** Run training first:
```bash
python scripts/train_model.py
```

---

### "Port 8000 already in use" (Backend won't start)

Another process is using that port.

**Solution:** Use a different port:
```bash
uvicorn backend.main:app --reload --port 8001
```

And update frontend `.env.local`:
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001
```

---

### Frontend shows "Failed to fetch"

Backend isn't running or wrong URL.

**Solution:**
1. Make sure backend is running (Terminal 1)
2. Check `frontend/.env.local` has correct URL
3. Try `http://127.0.0.1:8000` instead of `localhost:8000`

---

### "Dataset files not found"

PTB-XL dataset not downloaded or in wrong location.

**Solution:**
1. Download from https://physionet.org/content/ptb-xl/1.0.3/
2. Extract to `dataset/` folder
3. Check `dataset/ptbxl_database.csv` exists

---

### Windows: "Scripts\activate.ps1 cannot be loaded"

PowerShell execution policy blocks scripts.

**Solution:** Use Command Prompt (cmd) instead of PowerShell, or run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📊 Dataset Information

**PTB-XL v1.0.3** - The largest public 12-lead ECG dataset

- **Source:** PhysioNet / Computing in Cardiology
- **Size:** 21,837 clinical 12-lead ECGs
- **Duration:** 10 seconds per recording
- **Sampling Rates:** 100 Hz and 500 Hz
- **Patients:** 18,885 unique patients
- **Age Range:** 0-95 years
- **Diagnostic Labels:** 71 different statements mapped to 5 classes
- **License:** Open Database License (ODbL) v1.0

**Citation:**
```
Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I.,
Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available 
electrocardiography dataset. Scientific Data, 7(1), 154.
```

---

## ⚕️ Medical Disclaimer

> ⚠️ **IMPORTANT**: This is an AI-assisted analysis tool for **research and educational purposes ONLY**. 
>
> - It does NOT provide medical diagnoses
> - It is NOT a certified medical device
> - It should NOT replace professional medical judgment
> - Always consult a qualified healthcare professional for clinical decisions
>
> The developers assume no liability for medical decisions based on this software.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📄 License

This project is for educational and research purposes. The PTB-XL dataset is licensed under ODbL v1.0.

---

## 💡 Tips for Beginners

### Understanding the Pipeline

1. **Raw ECG → Image** - The model can't directly understand electrical signals, so we convert them to images (scalograms) that show frequency patterns over time.

2. **Vision Transformer** - Instead of using a pre-trained model (like ImageNet), we train from scratch because medical images are very different from everyday photos.

3. **Grad-CAM** - Shows "where the AI is looking" by highlighting important regions in the scalogram.

### Best Practices

- **Always activate the virtual environment** before running Python commands
- **Don't train on the full dataset initially** - use a smaller subset for testing (modify in `config.yaml`)
- **Monitor GPU usage** with `nvidia-smi` (if you have NVIDIA GPU)
- **Save your work** - training checkpoints are automatically saved, but back them up!
- **Read the logs** - check `outputs/*.log` files if something fails

### Learning Resources

- **Python Basics:** https://docs.python.org/3/tutorial/
- **PyTorch Tutorial:** https://pytorch.org/tutorials/
- **ECG Interpretation:** https://ecgwaves.com/
- **Vision Transformers:** https://arxiv.org/abs/2010.11929
- **Next.js Docs:** https://nextjs.org/docs

---

## 📞 Support

If you encounter issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review log files in `outputs/`
3. Make sure all setup steps were completed
4. Check that dataset files are in the correct location

---

**Happy analyzing! 🫀**
