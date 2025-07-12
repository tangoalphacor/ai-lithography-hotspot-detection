"""
Mock implementation of Hotspot Classifier without PyTorch dependencies
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union, Dict, Any
import cv2


class MockHotspotClassifier:
    """Mock hotspot classifier that simulates predictions without PyTorch."""
    
    def __init__(self):
        """Initialize the mock classifier."""
        self.models = ['ResNet18', 'ViT', 'EfficientNet']
        print("✅ Mock hotspot classification models loaded")
    
    def predict(self, image: Union[np.ndarray, Image.Image], 
                model_name: str = 'ResNet18') -> Tuple[str, float]:
        """
        Mock hotspot prediction based on image statistics.
        
        Args:
            image: Input image
            model_name: Name of the model to use
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Calculate image complexity metrics
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection for complexity analysis
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Statistical features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        contrast = std_intensity / (mean_intensity + 1e-6)
        
        # Model-specific biases
        model_bias = {
            'ResNet18': 0.1,
            'ViT': 0.15,
            'EfficientNet': 0.05
        }
        
        # Combine features for prediction
        complexity_score = edge_density * 2 + contrast * 0.01
        base_prob = np.clip(complexity_score + model_bias.get(model_name, 0.1), 0.1, 0.9)
        
        # Add some randomness for demo
        prob = base_prob + np.random.normal(0, 0.05)
        prob = np.clip(prob, 0.1, 0.9)
        
        prediction = "Hotspot" if prob > 0.5 else "No Hotspot"
        return prediction, float(prob)
    
    def get_feature_maps(self, image: Union[np.ndarray, Image.Image], 
                        model_name: str = 'ResNet18') -> np.ndarray:
        """Mock feature map extraction."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Simulate feature maps with random data
        if len(image.shape) >= 2:
            h, w = image.shape[:2]
            feature_h, feature_w = h // 32, w // 32
            feature_maps = np.random.random((1, 512, feature_h, feature_w))
            return feature_maps
        else:
            return np.random.random((1, 512, 7, 7))
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get mock model information."""
        model_specs = {
            'ResNet18': {'params': '11.7M', 'inference_ms': 15},
            'ViT': {'params': '86M', 'inference_ms': 45},
            'EfficientNet': {'params': '5.3M', 'inference_ms': 12}
        }
        
        spec = model_specs.get(model_name, {'params': 'Unknown', 'inference_ms': 0})
        
        return {
            'model_name': model_name,
            'total_parameters': spec['params'],
            'inference_time_ms': spec['inference_ms'],
            'device': 'CPU (mock)',
            'input_size': '224x224x3',
            'num_classes': 2
        }

# Create alias for compatibility  
HotspotClassifier = MockHotspotClassifier
