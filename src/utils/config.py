"""
CARDIOVISION - Configuration Manager
Loads and validates config.yaml, exposes typed configuration.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml


@dataclass
class DatasetConfig:
    root_dir: str = "./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    metadata_file: str = "ptbxl_database.csv"
    scp_statements_file: str = "scp_statements.csv"
    sampling_rate: int = 100
    records_dir_100: str = "records100"
    records_dir_500: str = "records500"


@dataclass
class LabelsConfig:
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "Normal", "Myocardial Infarction", "Arrhythmia",
        "Left Ventricular Hypertrophy", "ST/T Wave Abnormalities"
    ])
    priority_order: List[str] = field(default_factory=lambda: [
        "Myocardial Infarction", "Arrhythmia",
        "Left Ventricular Hypertrophy", "ST/T Wave Abnormalities", "Normal"
    ])


@dataclass
class SplittingConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    patient_level: bool = True


@dataclass
class SignalQualityConfig:
    max_nan_ratio: float = 0.01
    max_flat_ratio: float = 0.50
    amplitude_min_mv: float = -10.0
    amplitude_max_mv: float = 10.0


@dataclass
class ButterworthConfig:
    low_cutoff_hz: float = 0.5
    high_cutoff_hz: float = 40.0
    filter_order: int = 4


@dataclass
class PanTompkinsConfig:
    integration_window_ms: int = 150
    primary_lead: int = 1
    fallback_leads: List[int] = field(default_factory=lambda: [0, 5, 6])


@dataclass
class NormalizationConfig:
    range_min: float = 0.0
    range_max: float = 1.0


@dataclass
class PreprocessingConfig:
    signal_quality: SignalQualityConfig = field(default_factory=SignalQualityConfig)
    butterworth: ButterworthConfig = field(default_factory=ButterworthConfig)
    pan_tompkins: PanTompkinsConfig = field(default_factory=PanTompkinsConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class CWTConfig:
    wavelet: str = "morl"
    scales_start: int = 1
    scales_end: int = 128
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    output_dir: str = "data/processed/cwt_images"
    lead_mode: str = "composite"
    composite_layout: List[int] = field(default_factory=lambda: [3, 4])
    colormap: str = "jet"
    save_format: str = "png"


@dataclass
class ModelConfig:
    name: str = "CardioViT"
    input_size: List[int] = field(default_factory=lambda: [224, 224])
    input_channels: int = 3
    patch_size: int = 16
    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    num_classes: int = 5
    use_cls_token: bool = True
    use_positional_embedding: bool = True


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    name: str = "CosineAnnealingLR"
    T_max: int = 50
    eta_min: float = 0.000001


@dataclass
class LossConfig:
    name: str = "CrossEntropyLoss"
    use_class_weights: bool = True
    label_smoothing: float = 0.1


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: str = "val_loss"
    patience: int = 8
    min_delta: float = 0.001


@dataclass
class AugmentationConfig:
    enabled: bool = True
    time_shift_max: int = 10
    amplitude_scale_range: List[float] = field(default_factory=lambda: [0.95, 1.05])
    gaussian_noise_std: float = 0.005


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ReproducibilityConfig:
    seed: int = 42
    deterministic: bool = True


@dataclass
class DeviceConfig:
    use_cuda: bool = True
    fallback_to_cpu: bool = True


@dataclass
class OutputConfig:
    checkpoints_dir: str = "models/checkpoints"
    figures_dir: str = "outputs/figures"
    predictions_dir: str = "outputs/predictions"
    explanations_dir: str = "outputs/explanations"
    metrics_dir: str = "outputs/metrics"
    dataset_report: str = "outputs/dataset_report.json"
    training_log: str = "outputs/training_log.csv"
    splits_dir: str = "data/splits"


@dataclass
class ExplainabilityConfig:
    method: str = "gradcam_vit"
    target_layer: int = -1
    colormap: str = "jet"
    alpha: float = 0.5


@dataclass
class StreamlitConfig:
    page_title: str = "CardioVision"
    page_icon: str = "❤️"
    layout: str = "wide"
    disclaimer: str = (
        "⚠️ This is an AI-assisted analysis tool for research purposes only. "
        "It does not provide a definitive medical diagnosis. "
        "Always consult a qualified healthcare professional."
    )


@dataclass
class Config:
    """Root configuration dataclass."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cwt: CWTConfig = field(default_factory=CWTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)


def _build_dataclass(cls, data: dict):
    """Recursively build a dataclass from a dictionary."""
    if data is None:
        return cls()
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in fieldtypes:
            ft = cls.__dataclass_fields__[key]
            # Check if field type is itself a dataclass
            if hasattr(ft.type, '__dataclass_fields__') if isinstance(ft.type, type) else False:
                kwargs[key] = _build_dataclass(ft.type, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def _build_config_from_dict(data: dict) -> Config:
    """Build full Config from raw YAML dict, handling nested dataclasses."""
    cfg = Config()
    if data is None:
        return cfg

    mapping = {
        'dataset': (DatasetConfig, 'dataset'),
        'labels': (LabelsConfig, 'labels'),
        'splitting': (SplittingConfig, 'splitting'),
        'cwt': (CWTConfig, 'cwt'),
        'model': (ModelConfig, 'model'),
        'reproducibility': (ReproducibilityConfig, 'reproducibility'),
        'device': (DeviceConfig, 'device'),
        'output': (OutputConfig, 'output'),
        'explainability': (ExplainabilityConfig, 'explainability'),
        'streamlit': (StreamlitConfig, 'streamlit'),
    }

    for yaml_key, (dcls, attr_name) in mapping.items():
        if yaml_key in data:
            setattr(cfg, attr_name, _build_dataclass(dcls, data[yaml_key]))

    # Handle nested preprocessing
    if 'preprocessing' in data:
        pp = data['preprocessing']
        pre = PreprocessingConfig()
        if 'signal_quality' in pp:
            pre.signal_quality = _build_dataclass(SignalQualityConfig, pp['signal_quality'])
        if 'butterworth' in pp:
            pre.butterworth = _build_dataclass(ButterworthConfig, pp['butterworth'])
        if 'pan_tompkins' in pp:
            pre.pan_tompkins = _build_dataclass(PanTompkinsConfig, pp['pan_tompkins'])
        if 'normalization' in pp:
            pre.normalization = _build_dataclass(NormalizationConfig, pp['normalization'])
        cfg.preprocessing = pre

    # Handle nested training
    if 'training' in data:
        tr = data['training']
        train_cfg = TrainingConfig()
        for simple_key in ['epochs', 'batch_size', 'num_workers', 'mixed_precision', 'gradient_clip_norm']:
            if simple_key in tr:
                setattr(train_cfg, simple_key, tr[simple_key])
        if 'optimizer' in tr:
            train_cfg.optimizer = _build_dataclass(OptimizerConfig, tr['optimizer'])
        if 'scheduler' in tr:
            train_cfg.scheduler = _build_dataclass(SchedulerConfig, tr['scheduler'])
        if 'loss' in tr:
            train_cfg.loss = _build_dataclass(LossConfig, tr['loss'])
        if 'early_stopping' in tr:
            train_cfg.early_stopping = _build_dataclass(EarlyStoppingConfig, tr['early_stopping'])
        if 'augmentation' in tr:
            train_cfg.augmentation = _build_dataclass(AugmentationConfig, tr['augmentation'])
        cfg.training = train_cfg

    return cfg


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, searches in project root.

    Returns:
        Populated Config dataclass.
    """
    if config_path is None:
        # Search for config.yaml relative to project root
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).resolve().parents[2] / "config.yaml",
        ]
        for p in search_paths:
            if p.exists():
                config_path = str(p)
                break
        else:
            print("Warning: config.yaml not found, using defaults.")
            return Config()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    return _build_config_from_dict(raw)


def get_project_root() -> Path:
    """Get the project root directory (where config.yaml lives)."""
    # Walk up from this file: src/utils/config.py -> src/utils -> src -> root
    return Path(__file__).resolve().parents[2]


def get_device(cfg: Config) -> str:
    """Determine the compute device based on configuration."""
    import torch
    if cfg.device.use_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"
