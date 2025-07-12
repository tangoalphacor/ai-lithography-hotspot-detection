"""
Package initialization for models module.
"""

# Import mock implementations that work without heavy ML dependencies
from .cyclegan_mock import CycleGANProcessor
from .classifier_mock import HotspotClassifier  
from .gradcam_mock import GradCAMVisualizer

__all__ = ['CycleGANProcessor', 'HotspotClassifier', 'GradCAMVisualizer']
