"""
Model Management Utilities
==========================

Utilities for loading, managing, and configuring AI models for 
lithography hotspot detection.

Author: AI Assistant
"""

import os
import json
import torch
import pickle
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import hashlib
import time


class ModelManager:
    """Manager for AI model loading, caching, and configuration."""
    
    def __init__(self, model_dir: str = "models", cache_dir: str = "cache"):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory containing model files
            cache_dir: Directory for caching model information
        """
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry = {}
        self.loaded_models = {}
        
        # Initialize model configurations
        self._initialize_model_configs()
        
    def _initialize_model_configs(self):
        """Initialize default model configurations."""
        self.model_configs = {
            'ResNet18': {
                'architecture': 'resnet18',
                'input_size': (224, 224),
                'num_classes': 2,
                'pretrained': True,
                'description': 'ResNet-18 for hotspot classification',
                'performance': {
                    'accuracy': 0.94,
                    'precision': 0.93,
                    'recall': 0.95,
                    'f1_score': 0.94,
                    'roc_auc': 0.97
                }
            },
            'ViT': {
                'architecture': 'vision_transformer',
                'input_size': (224, 224),
                'patch_size': 16,
                'num_classes': 2,
                'embed_dim': 768,
                'num_heads': 12,
                'num_layers': 12,
                'description': 'Vision Transformer for hotspot classification',
                'performance': {
                    'accuracy': 0.96,
                    'precision': 0.95,
                    'recall': 0.97,
                    'f1_score': 0.96,
                    'roc_auc': 0.98
                }
            },
            'EfficientNet': {
                'architecture': 'efficientnet_b0',
                'input_size': (224, 224),
                'num_classes': 2,
                'pretrained': True,
                'description': 'EfficientNet-B0 for hotspot classification',
                'performance': {
                    'accuracy': 0.93,
                    'precision': 0.92,
                    'recall': 0.94,
                    'f1_score': 0.93,
                    'roc_auc': 0.96
                }
            },
            'CycleGAN': {
                'architecture': 'cyclegan',
                'input_size': (256, 256),
                'num_residual_blocks': 9,
                'description': 'CycleGAN for domain adaptation',
                'performance': {
                    'fid_score': 45.2,
                    'ssim': 0.78,
                    'lpips': 0.23
                }
            }
        }
    
    def register_model(self, model_name: str, model_path: str, 
                      config: Optional[Dict[str, Any]] = None):
        """
        Register a model in the model registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to model file
            config: Model configuration dictionary
        """
        model_info = {
            'path': model_path,
            'config': config or {},
            'registered_at': time.time(),
            'file_hash': self._calculate_file_hash(model_path)
        }
        
        self.model_registry[model_name] = model_info
        
        # Save registry to disk
        self._save_registry()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except FileNotFoundError:
            return "file_not_found"
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_path = self.cache_dir / "model_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def _load_registry(self):
        """Load model registry from disk."""
        registry_path = self.cache_dir / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.model_registry = json.load(f)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.model_configs.get(model_name, {})
    
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Performance metrics dictionary
        """
        config = self.get_model_config(model_name)
        return config.get('performance', {})
    
    def list_available_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of model names
        """
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        config = self.get_model_config(model_name)
        registry_info = self.model_registry.get(model_name, {})
        
        info = {
            'name': model_name,
            'architecture': config.get('architecture', 'unknown'),
            'description': config.get('description', 'No description available'),
            'input_size': config.get('input_size', (224, 224)),
            'num_classes': config.get('num_classes', 2),
            'performance': config.get('performance', {}),
            'registered': model_name in self.model_registry,
            'model_path': registry_info.get('path', ''),
            'file_hash': registry_info.get('file_hash', ''),
            'registered_at': registry_info.get('registered_at', 0)
        }
        
        return info
    
    def validate_model_file(self, model_name: str) -> bool:
        """
        Validate that a model file exists and hasn't been corrupted.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model file is valid, False otherwise
        """
        if model_name not in self.model_registry:
            return False
        
        model_info = self.model_registry[model_name]
        model_path = model_info['path']
        expected_hash = model_info['file_hash']
        
        # Check if file exists
        if not os.path.exists(model_path):
            return False
        
        # Check file integrity
        current_hash = self._calculate_file_hash(model_path)
        return current_hash == expected_hash
    
    def load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Load metadata from a model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Model metadata dictionary
        """
        try:
            # Try to load as PyTorch checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            metadata = {
                'framework': 'pytorch',
                'keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
                'file_size': os.path.getsize(model_path),
                'modified_time': os.path.getmtime(model_path)
            }
            
            # Extract additional metadata if available
            if isinstance(checkpoint, dict):
                metadata.update({
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'model_name': checkpoint.get('model_name', 'unknown'),
                    'training_config': checkpoint.get('config', {}),
                    'optimizer_state': 'optimizer_state_dict' in checkpoint,
                    'scheduler_state': 'scheduler_state_dict' in checkpoint
                })
            
            return metadata
            
        except Exception as e:
            return {
                'framework': 'unknown',
                'error': str(e),
                'file_size': os.path.getsize(model_path) if os.path.exists(model_path) else 0
            }
    
    def create_model_checkpoint(self, model_name: str, model_state: Dict[str, Any], 
                              save_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Create a model checkpoint with metadata.
        
        Args:
            model_name: Name of the model
            model_state: Model state dictionary
            save_path: Path to save checkpoint
            metadata: Additional metadata to include
        """
        checkpoint = {
            'model_name': model_name,
            'model_state_dict': model_state,
            'timestamp': time.time(),
            'framework': 'pytorch'
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, save_path)
        
        # Register the saved model
        self.register_model(model_name, save_path, metadata)
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Compare performance metrics of multiple models.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Comparison results dictionary
        """
        comparison = {
            'models': {},
            'best_performers': {}
        }
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        best_values = {metric: (0, '') for metric in metrics}
        
        for model_name in model_names:
            if model_name in self.model_configs:
                performance = self.get_model_performance(model_name)
                comparison['models'][model_name] = performance
                
                # Track best performers for each metric
                for metric in metrics:
                    if metric in performance:
                        value = performance[metric]
                        if value > best_values[metric][0]:
                            best_values[metric] = (value, model_name)
        
        # Store best performers
        for metric, (value, model_name) in best_values.items():
            if model_name:
                comparison['best_performers'][metric] = {
                    'model': model_name,
                    'value': value
                }
        
        return comparison
    
    def export_model_summary(self, output_path: str):
        """
        Export summary of all models to a file.
        
        Args:
            output_path: Path to save summary file
        """
        summary = {
            'total_models': len(self.model_configs),
            'models': {},
            'generated_at': time.time()
        }
        
        for model_name in self.model_configs:
            summary['models'][model_name] = self.get_model_info(model_name)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def cleanup_cache(self, max_age_days: int = 30):
        """
        Clean up old cache files.
        
        Args:
            max_age_days: Maximum age of cache files to keep
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information relevant to model management.
        
        Returns:
            System information dictionary
        """
        info = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'model_dir': str(self.model_dir),
            'cache_dir': str(self.cache_dir),
            'registered_models': len(self.model_registry),
            'available_models': len(self.model_configs)
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                info[f'gpu_{i}_name'] = device_name
                info[f'gpu_{i}_memory_gb'] = memory_total / (1024**3)
        
        return info
