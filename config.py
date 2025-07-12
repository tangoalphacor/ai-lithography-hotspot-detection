# Application Configuration
# =========================

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730"
    },
    "server": {
        "port": 8501,
        "enableCORS": False,
        "enableXsrfProtection": True
    }
}

# Model Configuration
MODEL_CONFIG = {
    "default_model": "ResNet18",
    "available_models": ["ResNet18", "ViT", "EfficientNet"],
    "model_paths": {
        "ResNet18": "models/resnet18_hotspot.pth",
        "ViT": "models/vit_hotspot.pth", 
        "EfficientNet": "models/efficientnet_hotspot.pth",
        "CycleGAN": "models/cyclegan_synthetic2sem.pth"
    },
    "input_size": (224, 224),
    "batch_size": 32,
    "num_classes": 2
}

# Image Processing Configuration  
IMAGE_CONFIG = {
    "supported_formats": [".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
    "max_file_size_mb": 10,
    "max_images_batch": 20,
    "resize_method": "lanczos",
    "normalize_range": (0, 1)
}

# CycleGAN Configuration
CYCLEGAN_CONFIG = {
    "input_size": (256, 256),
    "num_residual_blocks": 9,
    "lambda_cycle": 10.0,
    "lambda_identity": 5.0,
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999
}

# Grad-CAM Configuration
GRADCAM_CONFIG = {
    "target_layers": {
        "ResNet18": "layer4",
        "ViT": "transformer.layers.11",
        "EfficientNet": "features.7"
    },
    "colormap": "jet",
    "alpha_overlay": 0.4,
    "guided_gradcam": True
}

# UI Configuration
UI_CONFIG = {
    "sidebar_width": 300,
    "main_content_padding": "1rem",
    "color_palette": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "success": "#2ca02c",
        "warning": "#ff9800",
        "error": "#d62728",
        "info": "#17becf"
    },
    "chart_height": 400,
    "image_gallery_cols": 3
}

# Performance Metrics
PERFORMANCE_METRICS = {
    "ResNet18": {
        "accuracy": 0.94,
        "precision": 0.93,
        "recall": 0.95,
        "f1_score": 0.94,
        "roc_auc": 0.97,
        "parameters": "11.7M",
        "inference_time_ms": 15
    },
    "ViT": {
        "accuracy": 0.96,
        "precision": 0.95,
        "recall": 0.97,
        "f1_score": 0.96,
        "roc_auc": 0.98,
        "parameters": "86M",
        "inference_time_ms": 45
    },
    "EfficientNet": {
        "accuracy": 0.93,
        "precision": 0.92,
        "recall": 0.94,
        "f1_score": 0.93,
        "roc_auc": 0.96,
        "parameters": "5.3M",
        "inference_time_ms": 12
    }
}

# Dataset Information
DATASET_INFO = {
    "name": "Lithography Hotspot Dataset",
    "version": "v2.1",
    "total_images": 52000,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "synthetic_images": 30000,
    "real_sem_images": 22000,
    "hotspot_ratio": 0.35,
    "image_size": "224x224",
    "augmentations": [
        "rotation", "flip", "brightness", 
        "contrast", "gaussian_noise"
    ]
}

# API Configuration (for future extensions)
API_CONFIG = {
    "base_url": "https://api.lithography-detection.com",
    "version": "v1",
    "timeout": 30,
    "max_retries": 3,
    "rate_limit": 100  # requests per minute
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "logs/app.log",
    "max_file_size": "10MB",
    "backup_count": 5
}

# Cache Configuration
CACHE_CONFIG = {
    "model_cache_size": 3,  # Number of models to keep in memory
    "image_cache_size": 100,  # Number of processed images to cache
    "cache_ttl": 3600,  # Cache TTL in seconds
    "cache_dir": "cache"
}
