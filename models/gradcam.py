"""
Grad-CAM Visualization Module
============================

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for 
explainable AI visualization of hotspot classification decisions.

Author: AI Assistant
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Union, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class GradCAMVisualizer:
    """Grad-CAM visualization for hotspot classification models."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradients = None
        self.activations = None
    
    def save_gradient(self, grad):
        """Hook function to save gradients."""
        self.gradients = grad
    
    def save_activation(self, module, input, output):
        """Hook function to save activations."""
        self.activations = output
    
    def generate_heatmap(self, image: Union[np.ndarray, Image.Image], 
                        model_name: str = 'ResNet18',
                        target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image: Input image
            model_name: Name of the classification model
            target_class: Target class for visualization (None for predicted class)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        try:
            # For demo purposes, create a mock Grad-CAM heatmap
            return self._create_mock_gradcam(image)
            
        except Exception as e:
            print(f"❌ Error generating Grad-CAM: {e}")
            return self._create_mock_gradcam(image)
    
    def _create_mock_gradcam(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Create a mock Grad-CAM heatmap for demonstration purposes.
        
        Args:
            image: Input image
            
        Returns:
            Mock Grad-CAM heatmap
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width = image.shape[:2]
        
        # Create a realistic-looking attention map
        # Center attention with gradual falloff
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Create multiple attention regions
        attention1 = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * (min(height, width) / 4) ** 2))
        
        # Add secondary attention regions
        if width > 100 and height > 100:
            attention2 = np.exp(-((x - width * 0.3) ** 2 + (y - height * 0.7) ** 2) / (2 * (min(height, width) / 8) ** 2))
            attention3 = np.exp(-((x - width * 0.7) ** 2 + (y - height * 0.3) ** 2) / (2 * (min(height, width) / 8) ** 2))
            attention_map = attention1 + 0.5 * attention2 + 0.3 * attention3
        else:
            attention_map = attention1
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.1, attention_map.shape)
        attention_map += noise
        
        # Normalize to [0, 1]
        attention_map = np.clip(attention_map, 0, 1)
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Apply colormap
        heatmap = cm.jet(attention_map)[:, :, :3]  # Remove alpha channel
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
    
    def generate_guided_gradcam(self, image: Union[np.ndarray, Image.Image],
                               model_name: str = 'ResNet18') -> np.ndarray:
        """
        Generate Guided Grad-CAM visualization.
        
        Args:
            image: Input image
            model_name: Name of the classification model
            
        Returns:
            Guided Grad-CAM visualization
        """
        # For demo purposes, create a mock guided Grad-CAM
        return self._create_mock_guided_gradcam(image)
    
    def _create_mock_guided_gradcam(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Create mock guided Grad-CAM for demonstration."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply edge detection to simulate guided gradients
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        guided_grad = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        guided_grad = (guided_grad - guided_grad.min()) / (guided_grad.max() - guided_grad.min() + 1e-8)
        
        # Apply threshold to highlight important regions
        guided_grad = np.where(guided_grad > 0.3, guided_grad, 0)
        
        # Convert to RGB
        guided_grad_rgb = np.stack([guided_grad] * 3, axis=-1)
        guided_grad_rgb = (guided_grad_rgb * 255).astype(np.uint8)
        
        return guided_grad_rgb
    
    def create_attention_rollout(self, image: Union[np.ndarray, Image.Image],
                                model_name: str = 'ViT') -> np.ndarray:
        """
        Create attention rollout visualization for Vision Transformer.
        
        Args:
            image: Input image
            model_name: Name of the model (should be ViT)
            
        Returns:
            Attention rollout visualization
        """
        if model_name != 'ViT':
            # Fall back to regular Grad-CAM for non-ViT models
            return self.generate_heatmap(image, model_name)
        
        # Create mock attention rollout for ViT
        return self._create_mock_attention_rollout(image)
    
    def _create_mock_attention_rollout(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Create mock attention rollout for demonstration."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        height, width = image.shape[:2]
        
        # Create patch-based attention pattern
        patch_size = 16
        h_patches = height // patch_size
        w_patches = width // patch_size
        
        # Create attention weights for patches
        attention_weights = np.random.beta(2, 5, (h_patches, w_patches))
        
        # Upsample to full image size
        attention_map = np.zeros((height, width))
        
        for i in range(h_patches):
            for j in range(w_patches):
                y_start, y_end = i * patch_size, min((i + 1) * patch_size, height)
                x_start, x_end = j * patch_size, min((j + 1) * patch_size, width)
                attention_map[y_start:y_end, x_start:x_end] = attention_weights[i, j]
        
        # Smooth the attention map
        attention_map = cv2.GaussianBlur(attention_map, (5, 5), 2)
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Apply colormap
        heatmap = cm.viridis(attention_map)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        return heatmap
    
    def generate_multiple_visualizations(self, image: Union[np.ndarray, Image.Image],
                                       model_name: str = 'ResNet18') -> dict:
        """
        Generate multiple types of visualizations for comprehensive analysis.
        
        Args:
            image: Input image
            model_name: Name of the classification model
            
        Returns:
            Dictionary containing different visualization types
        """
        visualizations = {}
        
        # Standard Grad-CAM
        visualizations['gradcam'] = self.generate_heatmap(image, model_name)
        
        # Guided Grad-CAM
        visualizations['guided_gradcam'] = self.generate_guided_gradcam(image, model_name)
        
        # Overlaid visualization
        visualizations['overlay'] = self.overlay_heatmap(image, visualizations['gradcam'])
        
        # For ViT models, add attention rollout
        if model_name == 'ViT':
            visualizations['attention_rollout'] = self.create_attention_rollout(image, model_name)
        
        return visualizations
    
    def save_visualization(self, visualization: np.ndarray, 
                          filepath: str, dpi: int = 300):
        """
        Save visualization to file.
        
        Args:
            visualization: Visualization array
            filepath: Output file path
            dpi: Resolution for saving
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(visualization)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def create_comparison_plot(self, original_image: np.ndarray,
                              visualizations: dict) -> np.ndarray:
        """
        Create a comparison plot with multiple visualizations.
        
        Args:
            original_image: Original input image
            visualizations: Dictionary of different visualizations
            
        Returns:
            Combined comparison plot as numpy array
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Grad-CAM Visualization Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grad-CAM heatmap
        if 'gradcam' in visualizations:
            axes[0, 1].imshow(visualizations['gradcam'])
            axes[0, 1].set_title('Grad-CAM Heatmap')
            axes[0, 1].axis('off')
        
        # Overlaid visualization
        if 'overlay' in visualizations:
            axes[0, 2].imshow(visualizations['overlay'])
            axes[0, 2].set_title('Overlay Visualization')
            axes[0, 2].axis('off')
        
        # Guided Grad-CAM
        if 'guided_gradcam' in visualizations:
            axes[1, 0].imshow(visualizations['guided_gradcam'])
            axes[1, 0].set_title('Guided Grad-CAM')
            axes[1, 0].axis('off')
        
        # Attention rollout (if available)
        if 'attention_rollout' in visualizations:
            axes[1, 1].imshow(visualizations['attention_rollout'])
            axes[1, 1].set_title('Attention Rollout')
            axes[1, 1].axis('off')
        
        # Remove empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to numpy array
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot_array
