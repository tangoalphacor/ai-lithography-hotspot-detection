"""
Test script to verify About page functionality
"""
import sys
import os
from pathlib import Path

def test_about_page():
    """Test if the About page can be imported and basic functions work"""
    
    print("🧪 Testing About Page Functionality")
    print("=" * 50)
    
    # Test 1: Import check
    print("\n1. Testing imports...")
    try:
        from pages.about import show_about_page, create_download_link, create_test_images_zip
        print("   ✅ Successfully imported About page functions")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Check if test images exist
    print("\n2. Testing test images...")
    test_images_dir = Path("test_images")
    if test_images_dir.exists():
        image_count = len(list(test_images_dir.glob("*")))
        print(f"   ✅ Found {image_count} test images in test_images/")
    else:
        print("   ⚠️  Test images directory not found")
        print("   💡 Run: python create_test_images.py")
    
    # Test 3: Test download link creation
    print("\n3. Testing download functionality...")
    try:
        if test_images_dir.exists():
            test_files = list(test_images_dir.glob("*.jpg"))
            if test_files:
                link = create_download_link(test_files[0], "Test Download")
                if "<a href=" in link:
                    print("   ✅ Download link creation working")
                else:
                    print("   ❌ Download link creation failed")
        else:
            print("   ⚠️  Skipped - no test images found")
    except Exception as e:
        print(f"   ❌ Download test failed: {e}")
    
    # Test 4: Test ZIP creation
    print("\n4. Testing ZIP creation...")
    try:
        zip_link = create_test_images_zip()
        if "Download All Test Images" in zip_link:
            print("   ✅ ZIP creation working")
        else:
            print("   ❌ ZIP creation failed")
    except Exception as e:
        print(f"   ❌ ZIP test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ About page testing complete!")
    print("\n🚀 To test the About page:")
    print("   1. Run: streamlit run demo_about.py")
    print("   2. Or use navigation in main app")
    print("   3. Check all sections load properly")
    
    return True

if __name__ == "__main__":
    test_about_page()
