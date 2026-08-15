"""FastAPI backend for CardioVision model inference."""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import CardiovisionPredictor
from src.models.vit import CardioViT
from src.utils.config import get_device, load_config

app = FastAPI(
    title="CardioVision API",
    description="Backend server for ECG upload, CardioViT inference, and explainability output.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
_predictor: CardiovisionPredictor | None = None
_model_metadata: dict[str, Any] | None = None


def _load_predictor() -> CardiovisionPredictor:
    global _predictor, _model_metadata
    if _predictor is not None:
        return _predictor

    device = get_device(cfg)
    model = CardioViT(
        image_size=cfg.model.input_size[0],
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.input_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embedding_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        use_cls_token=cfg.model.use_cls_token,
        use_positional_embedding=cfg.model.use_positional_embedding,
    )

    checkpoint_path = Path(cfg.output.checkpoints_dir) / "best_model.pth"
    checkpoint_loaded = False
    checkpoint_epoch = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        checkpoint_loaded = True
        checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    _model_metadata = {
        "name": cfg.model.name,
        "version": "CardioViT-v1",
        "device": device,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "class_names": cfg.labels.class_names,
    }
    _predictor = CardiovisionPredictor(model, cfg, device)
    return _predictor


def _array_to_png_data_url(array: np.ndarray, *, chw: bool = False) -> str:
    image_array = array.transpose(1, 2, 0) if chw else array
    image_array = np.nan_to_num(image_array)
    if image_array.max(initial=0) <= 1.0:
        image_array = image_array * 255
    image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _downsample_signal(signal: np.ndarray, max_points: int = 800) -> list[dict[str, float]]:
    if signal.ndim != 2 or signal.shape[0] == 0:
        return []
    lead_idx = min(cfg.preprocessing.pan_tompkins.primary_lead, signal.shape[1] - 1)
    lead_signal = signal[:, lead_idx]
    if len(lead_signal) > max_points:
        indices = np.linspace(0, len(lead_signal) - 1, max_points).astype(int)
        lead_signal = lead_signal[indices]
    else:
        indices = np.arange(len(lead_signal))
    return [
        {"x": float(idx), "y": float(value)}
        for idx, value in zip(indices, np.nan_to_num(lead_signal), strict=False)
    ]


def _risk_level(predicted_class: str, confidence: float) -> str:
    if predicted_class == "Normal":
        return "Low" if confidence >= 0.60 else "Review"
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.50:
        return "Moderate"
    return "Review"


def _clinical_summary(predicted_class: str, confidence_pct: float) -> str:
    if predicted_class == "Normal":
        return f"The model found the strongest evidence for a normal ECG pattern with {confidence_pct:.1f}% confidence."
    return (
        f"The model found the strongest evidence for {predicted_class} with "
        f"{confidence_pct:.1f}% confidence. This output is intended for research support only."
    )


def _serialize_prediction(results: dict[str, Any], record_id: str, elapsed_seconds: float) -> dict[str, Any]:
    if results.get("status") != "success":
        raise HTTPException(status_code=422, detail=results.get("message", "Inference failed"))

    predicted_class = str(results["predicted_class_name"])
    confidence = float(results["confidence"])
    confidence_pct = round(confidence * 100, 2)
    probabilities = [
        {"condition": name, "probability": round(float(prob) * 100, 2)}
        for name, prob in results["probabilities"].items()
    ]
    probabilities.sort(key=lambda item: item["probability"], reverse=True)

    rr_intervals = np.asarray(results.get("rr_intervals", []), dtype=float)
    heart_rate = None
    if rr_intervals.size > 0:
        heart_rate = round(float(60 * cfg.dataset.sampling_rate / np.mean(rr_intervals)), 1)

    payload = {
        "id": record_id,
        "timestamp": time.strftime("%B %d, %Y at %I:%M %p"),
        "predictedClass": predicted_class,
        "confidence": confidence_pct,
        "riskLevel": _risk_level(predicted_class, confidence),
        "probabilities": probabilities,
        "clinicalSummary": _clinical_summary(predicted_class, confidence_pct),
        "recommendation": cfg.streamlit.disclaimer,
        "inferenceTime": round(elapsed_seconds, 2),
        "modelVersion": (_model_metadata or {}).get("version", "CardioViT-v1"),
        "heartRateBpm": heart_rate,
        "rPeakCount": int(len(results.get("r_peaks", []))),
        "signalPreview": _downsample_signal(results["filtered_signal"]),
        "scalogramImage": _array_to_png_data_url(results["scalogram_chw"], chw=True),
        "gradcamImage": _array_to_png_data_url(results["gradcam_overlay"], chw=False)
        if "gradcam_overlay" in results
        else None,
    }
    return payload


def _find_sample_record() -> Path:
    records_dir = Path(cfg.dataset.root_dir) / cfg.dataset.records_dir_100
    sample = next(records_dir.rglob("*.hea"), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="No sample WFDB record found in dataset.")
    return sample.with_suffix("")


@app.on_event("startup")
def startup() -> None:
    _load_predictor()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    _load_predictor()
    metrics_path = Path(cfg.output.metrics_dir) / "test_metrics.json"
    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"model": _model_metadata, "metrics": metrics}


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    dat_file = next((file for file in files if file.filename and file.filename.endswith(".dat")), None)
    hea_file = next((file for file in files if file.filename and file.filename.endswith(".hea")), None)
    if dat_file is None or hea_file is None:
        raise HTTPException(status_code=400, detail="Upload both matching .dat and .hea WFDB files.")

    predictor = _load_predictor()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dat_path = tmp_path / Path(dat_file.filename).name
        hea_path = tmp_path / Path(hea_file.filename).name
        dat_path.write_bytes(await dat_file.read())
        hea_path.write_bytes(await hea_file.read())
        record_path = tmp_path / hea_path.stem
        results = predictor.predict_from_file(str(record_path), generate_explanation=True)

    return _serialize_prediction(results, hea_path.stem, time.perf_counter() - start)


@app.post("/api/sample")
def analyze_sample() -> dict[str, Any]:
    predictor = _load_predictor()
    record_path = _find_sample_record()
    start = time.perf_counter()
    results = predictor.predict_from_file(str(record_path), generate_explanation=True)
    return _serialize_prediction(results, record_path.name, time.perf_counter() - start)
