"""
Mock implementation of Grad-CAM without PyTorch dependencies
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Union, Dict, Any


class MockGradCAMVisualizer:
    """Mock Grad-CAM visualizer that creates realistic heatmaps without PyTorch."""
    
    def __init__(self):
        """Initialize the mock visualizer."""
        print("✅ Mock Grad-CAM visualizer initialized")
    
    def generate_heatmap(self, image: Union[np.ndarray, Image.Image], 
                        model_name: str = 'ResNet18') -> np.ndarray:
        """
        Generate a realistic mock Grad-CAM heatmap.
        
        Args:
            image: Input image
            model_name: Name of the classification model
            
        Returns:
            Mock Grad-CAM heatmap as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        height, width = image.shape[:2]
        
        # Create realistic attention patterns based on image features
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Use edge detection to find interesting regions
        edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
        
        # Create attention map based on edges
        attention_base = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
        
        # Add multiple attention regions
        y, x = np.ogrid[:height, :width]
        
        # Primary attention (center-biased)
        center_y, center_x = height // 2, width // 2
        center_attention = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * (min(height, width) / 3) ** 2))
        
        # Secondary attention regions based on high-gradient areas
        if np.max(attention_base) > 0:
            attention_base = attention_base / np.max(attention_base)
        
        # Combine different attention sources
        combined_attention = 0.6 * center_attention + 0.4 * attention_base
        
        # Add some model-specific characteristics
        if model_name == 'ViT':
            # Patch-based attention for Vision Transformer
            patch_size = 16
            for i in range(0, height, patch_size):
                for j in range(0, width, patch_size):
                    if np.random.random() > 0.7:  # Random patches get attention
                        combined_attention[i:i+patch_size, j:j+patch_size] *= 1.5
        
        # Normalize and add noise
        combined_attention = np.clip(combined_attention, 0, 1)
        noise = np.random.normal(0, 0.05, combined_attention.shape)
        combined_attention = np.clip(combined_attention + noise, 0, 1)
        
        # Normalize to [0, 1]
        if np.max(combined_attention) > 0:
            combined_attention = combined_attention / np.max(combined_attention)
        
        # Apply colormap
        heatmap = cm.jet(combined_attention)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        return heatmap
    
    def overlay_heatmap(self, image: Union[np.ndarray, Image.Image], 
                       heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            alpha: Transparency factor for overlay
            
        Returns:
            Image with overlaid heatmap
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure same dimensions
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Overlay heatmap
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def generate_multiple_visualizations(self, image: Union[np.ndarray, Image.Image],
                                       model_name: str = 'ResNet18') -> Dict[str, Any]:
        """
        Generate multiple visualization types.
        
        Args:
            image: Input image
            model_name: Name of the classification model
            
        Returns:
            Dictionary containing different visualization types
        """
        visualizations = {}
        
        # Standard Grad-CAM
        visualizations['gradcam'] = self.generate_heatmap(image, model_name)
        
        # Guided Grad-CAM (simplified)
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Edge-based guided gradients
        guided_grad = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 1, ksize=3)
        guided_grad = np.abs(guided_grad)
        guided_grad = (guided_grad / np.max(guided_grad) * 255).astype(np.uint8)
        guided_grad_rgb = cv2.cvtColor(guided_grad, cv2.COLOR_GRAY2RGB)
        visualizations['guided_gradcam'] = guided_grad_rgb
        
        # Overlaid visualization
        visualizations['overlay'] = self.overlay_heatmap(image, visualizations['gradcam'])
        
        return visualizations

# Create alias for compatibility
GradCAMVisualizer = MockGradCAMVisualizer
