"""
Mock implementation of CycleGAN without PyTorch dependencies
"""

import numpy as np
from PIL import Image
import cv2
from typing import Union, Optional


class MockCycleGANProcessor:
    """Mock CycleGAN processor that simulates domain adaptation without PyTorch."""
    
    def __init__(self):
        """Initialize the mock processor."""
        print("✅ Mock CycleGAN processor initialized")
    
    def translate(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Create a mock SEM-style translation.
        
        Args:
            image: Input synthetic lithography image
            
        Returns:
            Mock translated SEM-style image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()
        
        # Create SEM-like appearance with noise and texture
        noise = np.random.normal(0, 8, gray.shape)
        sem_like = np.clip(gray * 0.85 + noise * 0.15, 0, 255)
        
        # Apply slight blur to simulate SEM characteristics
        sem_like = cv2.GaussianBlur(sem_like.astype(np.uint8), (3, 3), 0.5)
        
        # Convert back to RGB with slight blue tint
        sem_rgb = np.stack([sem_like, sem_like, np.clip(sem_like * 1.1, 0, 255)], axis=-1)
        
        return sem_rgb.astype(np.uint8)
    
    def get_model_info(self) -> dict:
        """Get mock model information."""
        return {
            'model_type': 'Mock CycleGAN Generator',
            'total_parameters': '11.4M (simulated)',
            'device': 'CPU (mock)',
            'input_size': '256x256x3',
            'output_size': '256x256x3'
        }

# Create alias for compatibility
CycleGANProcessor = MockCycleGANProcessor
