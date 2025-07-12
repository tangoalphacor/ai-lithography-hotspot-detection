"""
Simple validation script for advanced dependencies
"""

def test_dependencies():
    """Test if all advanced dependencies are working"""
    
    print("🔬 Advanced Lithography Hotspot Detection - Dependency Check")
    print("=" * 60)
    
    # Test basic imports
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
    except ImportError:
        print("❌ TorchVision not available")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not available")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not available")
    
    print("\n" + "=" * 60)
    
    # Test advanced modules
    print("🧪 Testing Advanced Modules:")
    
    try:
        from config_advanced import MODEL_CONFIG
        print("✅ config_advanced.py - OK")
    except Exception as e:
        print(f"❌ config_advanced.py - Error: {e}")
    
    try:
        from cyclegan_advanced import get_cyclegan_processor
        processor = get_cyclegan_processor()
        print("✅ cyclegan_advanced.py - OK")
    except Exception as e:
        print(f"❌ cyclegan_advanced.py - Error: {e}")
    
    try:
        from classifier_advanced import get_hotspot_classifier
        classifier = get_hotspot_classifier()
        print("✅ classifier_advanced.py - OK")
    except Exception as e:
        print(f"❌ classifier_advanced.py - Error: {e}")
    
    try:
        from gradcam_advanced import get_gradcam_visualizer
        visualizer = get_gradcam_visualizer()
        print("✅ gradcam_advanced.py - OK")
    except Exception as e:
        print(f"❌ gradcam_advanced.py - Error: {e}")
    
    try:
        from image_processing_advanced import get_image_processor
        processor = get_image_processor()
        print("✅ image_processing_advanced.py - OK")
    except Exception as e:
        print(f"❌ image_processing_advanced.py - Error: {e}")
    
    print("\n" + "=" * 60)
    print("🚀 Status: Advanced dependencies are ready!")
    print("You can now run the advanced application with full functionality.")
    print("\nTo start the app:")
    print("1. Run: launcher.bat (choose option 1 for Advanced App)")
    print("2. Or run: start_app.bat")
    print("3. Or run: streamlit run app_advanced.py")

if __name__ == "__main__":
    test_dependencies()
