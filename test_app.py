"""
Automated testing script for the Streamlit Lithography Hotspot Detection app
This script tests the core functionality programmatically
"""

import requests
import time
import os
import sys
from pathlib import Path

def test_streamlit_app():
    """Test if Streamlit app is running and responsive"""
    
    print("🧪 Automated Testing for Lithography Hotspot Detection App")
    print("=" * 60)
    
    # Test 1: Check if app is running
    print("\n1. Testing App Availability...")
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("   ✅ App is running successfully")
        else:
            print(f"   ❌ App returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to app: {e}")
        print("   💡 Make sure Streamlit is running on http://localhost:8501")
        return False
    
    # Test 2: Check if test images exist
    print("\n2. Testing Test Images...")
    test_images_dir = Path("test_images")
    expected_images = [
        "small_test.jpg",
        "medium_test.png", 
        "large_test.jpg",
        "wide_test.png",
        "tall_test.jpg"
    ]
    
    missing_images = []
    for img in expected_images:
        img_path = test_images_dir / img
        if img_path.exists():
            print(f"   ✅ Found: {img}")
        else:
            print(f"   ❌ Missing: {img}")
            missing_images.append(img)
    
    if missing_images:
        print(f"\n   💡 Run: python create_test_images.py to create missing images")
        return False
    
    # Test 3: Check if required modules can be imported
    print("\n3. Testing Python Dependencies...")
    required_modules = [
        "streamlit",
        "numpy", 
        "pandas",
        "PIL",
        "matplotlib",
        "plotly",
        "cv2"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            if module == "PIL":
                from PIL import Image
                print(f"   ✅ {module} (Pillow) imported successfully")
            elif module == "cv2":
                import cv2
                print(f"   ✅ {module} (OpenCV) imported successfully")
            else:
                __import__(module)
                print(f"   ✅ {module} imported successfully")
        except ImportError as e:
            print(f"   ❌ {module} import failed: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n   💡 Install missing modules: pip install {' '.join(failed_imports)}")
        return False
    
    # Test 4: Check app files
    print("\n4. Testing App Files...")
    required_files = [
        "app_working.py",
        "config.py",
        "requirements.txt",
        "models/cyclegan_mock.py",
        "models/classifier_mock.py", 
        "models/gradcam_mock.py",
        "utils/image_processing.py",
        "utils/model_utils.py",
        "utils/ui_components.py"
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ Found: {file}")
        else:
            print(f"   ❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n   💡 Some app files are missing. Check project structure.")
        return False
    
    # Test 5: Performance check
    print("\n5. Testing Performance...")
    start_time = time.time()
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        response_time = time.time() - start_time
        if response_time < 3.0:
            print(f"   ✅ App responds in {response_time:.2f} seconds")
        else:
            print(f"   ⚠️  App responds slowly: {response_time:.2f} seconds")
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All automated tests passed!")
    print("\n📋 Manual Testing Checklist:")
    print("   1. Open http://localhost:8501 in your browser")
    print("   2. Upload test images from test_images/ folder")  
    print("   3. Test domain adaptation feature")
    print("   4. Test hotspot classification with different models")
    print("   5. Test Grad-CAM visualization")
    print("   6. Test batch processing")
    print("   7. Test download functionality")
    print("   8. Test theme toggle (light/dark)")
    print("\n🚀 Happy Testing!")
    
    return True

def main():
    """Main testing function"""
    success = test_streamlit_app()
    if not success:
        print("\n❌ Some tests failed. Please fix issues before manual testing.")
        sys.exit(1)
    else:
        print("\n✅ Ready for manual testing!")
        sys.exit(0)

if __name__ == "__main__":
    main()
