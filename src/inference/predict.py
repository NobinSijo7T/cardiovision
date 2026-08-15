"""
CARDIOVISION - Inference Pipeline
End-to-end prediction from raw ECG file to class probabilities and explanation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wfdb
from PIL import Image

from src.utils.logger import get_logger
from src.preprocessing.signal_quality import check_signal_quality
from src.preprocessing.butterworth import apply_butterworth_filter
from src.preprocessing.pan_tompkins import detect_r_peaks
from src.preprocessing.normalization import min_max_normalize
from src.preprocessing.cwt_transform import generate_composite_scalogram
from src.explainability.gradcam_vit import ViTGradCAM, overlay_heatmap

logger = get_logger("cardiovision.inference")

class CardiovisionPredictor:
    """End-to-end prediction pipeline."""
    
    def __init__(self, model: torch.nn.Module, cfg, device: str = "cpu"):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.grad_cam = ViTGradCAM(self.model, target_layer_idx=self.cfg.explainability.target_layer)
        
        self.class_names = self.cfg.labels.class_names
        
    def predict_from_signal(
        self, 
        signal: np.ndarray, 
        sampling_rate: Optional[int] = None,
        generate_explanation: bool = True
    ) -> Dict:
        """
        Run inference on a raw ECG signal array.
        
        Args:
            signal: ECG array (num_samples, num_leads).
            sampling_rate: Sampling rate in Hz (default: from config).
            generate_explanation: Whether to generate Grad-CAM heatmap.
            
        Returns:
            Dictionary with prediction results and intermediate data.
        """
        sr = sampling_rate or self.cfg.dataset.sampling_rate
        
        result = {
            "status": "success",
            "message": "",
            "raw_signal": signal.copy()
        }
        
        # 1. Quality Check
        is_valid, issues = check_signal_quality(
            signal,
            max_nan_ratio=self.cfg.preprocessing.signal_quality.max_nan_ratio,
            max_flat_ratio=self.cfg.preprocessing.signal_quality.max_flat_ratio,
            amplitude_min_mv=self.cfg.preprocessing.signal_quality.amplitude_min_mv,
            amplitude_max_mv=self.cfg.preprocessing.signal_quality.amplitude_max_mv
        )
        if not is_valid:
            result["status"] = "error"
            result["message"] = f"Signal quality check failed: {', '.join(issues)}"
            return result
            
        # 2. Butterworth Filter
        filtered_signal = apply_butterworth_filter(
            signal, sr,
            self.cfg.preprocessing.butterworth.low_cutoff_hz,
            self.cfg.preprocessing.butterworth.high_cutoff_hz,
            self.cfg.preprocessing.butterworth.filter_order
        )
        result["filtered_signal"] = filtered_signal
        
        # 3. Pan-Tompkins R-Peaks (for visualization only, not used by ViT directly)
        r_peaks, rr_intervals = detect_r_peaks(
            filtered_signal, sr,
            primary_lead=self.cfg.preprocessing.pan_tompkins.primary_lead,
            fallback_leads=self.cfg.preprocessing.pan_tompkins.fallback_leads,
            integration_window_ms=self.cfg.preprocessing.pan_tompkins.integration_window_ms
        )
        result["r_peaks"] = r_peaks
        result["rr_intervals"] = rr_intervals
        
        # 4. Normalize
        normalized_signal = min_max_normalize(
            filtered_signal,
            self.cfg.preprocessing.normalization.range_min,
            self.cfg.preprocessing.normalization.range_max
        )
        
        # 5. CWT Scalogram
        scalogram = generate_composite_scalogram(
            normalized_signal, sr,
            wavelet=self.cfg.cwt.wavelet,
            scales_start=self.cfg.cwt.scales_start,
            scales_end=self.cfg.cwt.scales_end,
            image_size=tuple(self.cfg.cwt.image_size),
            colormap=self.cfg.cwt.colormap,
            layout=tuple(self.cfg.cwt.composite_layout)
        )
        # scalogram is CHW, [0, 1]
        result["scalogram_chw"] = scalogram
        
        # 6. Model Inference
        input_tensor = torch.from_numpy(scalogram).unsqueeze(0).to(self.device) # (1, C, H, W)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = probs[pred_idx]
        
        result["predicted_class_idx"] = pred_idx
        result["predicted_class_name"] = pred_class
        result["confidence"] = float(confidence)
        result["probabilities"] = {name: float(p) for name, p in zip(self.class_names, probs)}
        
        # 7. Explainability
        if generate_explanation:
            heatmap = self.grad_cam(input_tensor, target_class=pred_idx)
            
            # Convert CHW scalogram to HWC for overlay
            img_hwc = scalogram.transpose(1, 2, 0)
            overlay = overlay_heatmap(
                img_hwc, heatmap, 
                alpha=self.cfg.explainability.alpha
            )
            
            result["gradcam_heatmap"] = heatmap
            result["gradcam_overlay"] = overlay
            
        return result
        
    def predict_from_file(self, record_path: str, generate_explanation: bool = True) -> Dict:
        """
        Run inference on a WFDB record file.
        
        Args:
            record_path: Path to record (without .dat/.hea extension).
            
        Returns:
            Dictionary with prediction results.
        """
        try:
            signal, meta = wfdb.rdsamp(record_path)
            sr = meta['fs']
            return self.predict_from_signal(signal, sr, generate_explanation)
        except Exception as e:
            logger.error(f"Failed to load or process record {record_path}: {e}")
            return {
                "status": "error",
                "message": f"Failed to load or process record: {e}"
            }

    def predict_from_image(self, image_path: str, generate_explanation: bool = True) -> Dict:
        """
        Run inference directly on a pre-generated CWT/scalogram image.

        Args:
            image_path: Path to a PNG/JPG/JPEG image.
            generate_explanation: Whether to generate Grad-CAM heatmap.

        Returns:
            Dictionary with prediction results and image-based intermediate data.
        """
        try:
            image_size = tuple(self.cfg.model.input_size)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((image_size[1], image_size[0]))
            scalogram = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0

            result = {
                "status": "success",
                "message": "",
                "input_type": "image",
                "source_name": Path(image_path).name,
                "scalogram_chw": scalogram,
            }

            input_tensor = torch.from_numpy(scalogram).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_class = self.class_names[pred_idx]
            confidence = probs[pred_idx]

            result["predicted_class_idx"] = pred_idx
            result["predicted_class_name"] = pred_class
            result["confidence"] = float(confidence)
            result["probabilities"] = {name: float(p) for name, p in zip(self.class_names, probs)}

            if generate_explanation:
                heatmap = self.grad_cam(input_tensor, target_class=pred_idx)
                img_hwc = scalogram.transpose(1, 2, 0)
                overlay = overlay_heatmap(
                    img_hwc,
                    heatmap,
                    alpha=self.cfg.explainability.alpha,
                )

                result["gradcam_heatmap"] = heatmap
                result["gradcam_overlay"] = overlay

            return result
        except Exception as e:
            logger.error(f"Failed to load or process image {image_path}: {e}")
            return {
                "status": "error",
                "message": f"Failed to load or process image: {e}",
            }
