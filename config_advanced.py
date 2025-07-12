"""
Advanced Configuration for AI-based Lithography Hotspot Detection
================================================================

This module contains all advanced configuration parameters for the production-ready
lithography hotspot detection system with real AI models.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # CycleGAN Configuration
    cyclegan_checkpoint: str = "models/checkpoints/cyclegan_latest.pth"
    cyclegan_input_size: Tuple[int, int] = (256, 256)
    cyclegan_batch_size: int = 1
    cyclegan_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Classification Models Configuration
    resnet_checkpoint: str = "models/checkpoints/resnet18_hotspot.pth"
    vit_checkpoint: str = "models/checkpoints/vit_hotspot.pth"
    efficientnet_checkpoint: str = "models/checkpoints/efficientnet_hotspot.pth"
    
    # Model Parameters
    num_classes: int = 2  # Hotspot / No Hotspot
    image_size: Tuple[int, int] = (224, 224)
    classification_batch_size: int = 16
    
    # Grad-CAM Configuration
    gradcam_target_layers: Dict[str, List[str]] = field(default_factory=lambda: {
        "resnet18": ["layer4"],
        "vit": ["blocks.11.norm1"],
        "efficientnet": ["features.8"]
    })

@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Supported formats
    supported_formats: List[str] = None
    max_file_size_mb: int = 50
    min_image_size: Tuple[int, int] = (64, 64)
    max_image_size: Tuple[int, int] = (2048, 2048)
    
    # Preprocessing
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data augmentation for training
    augmentation_params: Dict = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
        
        if self.augmentation_params is None:
            self.augmentation_params = {
                'rotation_range': 30,
                'brightness_range': [0.8, 1.2],
                'contrast_range': [0.8, 1.2],
                'noise_factor': 0.1
            }

@dataclass
class AppConfig:
    """Main application configuration"""
    # App metadata
    app_name: str = "Advanced AI Lithography Hotspot Detection"
    version: str = "2.0.0"
    author: str = "Abhinav"
    
    # Paths
    base_dir: Path = Path(__file__).parent
    models_dir: Path = base_dir / "models"
    checkpoints_dir: Path = models_dir / "checkpoints"
    assets_dir: Path = base_dir / "assets"
    temp_dir: Path = base_dir / "temp"
    results_dir: Path = base_dir / "results"
    
    # Performance settings
    enable_gpu: bool = torch.cuda.is_available()
    max_concurrent_processes: int = 4
    cache_size_mb: int = 1024
    
    # UI Configuration
    theme_colors: Dict[str, str] = None
    custom_css_file: str = "assets/advanced_styles.css"
    
    # API Configuration
    enable_api: bool = True
    api_host: str = "localhost"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    def __post_init__(self):
        # Create directories
        for dir_path in [self.models_dir, self.checkpoints_dir, self.assets_dir, 
                        self.temp_dir, self.results_dir, Path("logs")]:
            dir_path.mkdir(exist_ok=True)
        
        if self.theme_colors is None:
            self.theme_colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'warning': '#d62728',
                'info': '#17becf',
                'background': '#0e1117',
                'surface': '#262730'
            }

# Global configuration instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
APP_CONFIG = AppConfig()

# Model URLs for downloading pre-trained models
MODEL_URLS = {
    'cyclegan': 'https://github.com/user/models/releases/download/v1.0/cyclegan_synthetic_sem.pth',
    'resnet18': 'https://github.com/user/models/releases/download/v1.0/resnet18_hotspot_detector.pth',
    'vit': 'https://github.com/user/models/releases/download/v1.0/vit_hotspot_detector.pth',
    'efficientnet': 'https://github.com/user/models/releases/download/v1.0/efficientnet_hotspot_detector.pth'
}

# Advanced processing parameters
PROCESSING_CONFIG = {
    'preprocessing': {
        'gaussian_blur_kernel': 3,
        'noise_reduction_iterations': 2,
        'contrast_enhancement': True,
        'histogram_equalization': True
    },
    'postprocessing': {
        'confidence_smoothing': True,
        'spatial_consistency_check': True,
        'temporal_consistency_frames': 5
    },
    'batch_processing': {
        'chunk_size': 32,
        'parallel_workers': 4,
        'memory_limit_gb': 8
    }
}

# Evaluation metrics configuration
METRICS_CONFIG = {
    'classification_metrics': [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'auc_roc', 'confusion_matrix'
    ],
    'detection_metrics': [
        'mean_average_precision', 'intersection_over_union',
        'precision_at_recall', 'area_under_curve'
    ],
    'performance_metrics': [
        'inference_time', 'memory_usage', 'gpu_utilization',
        'throughput_fps'
    ]
}
