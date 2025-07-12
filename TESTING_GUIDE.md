# Streamlit App Testing Guide

## 🧪 Comprehensive Testing Checklist

### 1. **Basic UI Testing**
- [ ] **App Launch**: Verify the app loads without errors at `http://localhost:8501`
- [ ] **Theme Toggle**: Test light/dark theme switching in the sidebar
- [ ] **Responsive Design**: Check layout on different screen sizes
- [ ] **Navigation**: Verify all sidebar sections are accessible

### 2. **File Upload Testing**
#### Test Cases:
- [ ] **Valid Image Upload**: Upload `.jpg`, `.png`, `.jpeg` files
- [ ] **Invalid File Type**: Try uploading `.txt`, `.pdf` files (should show error)
- [ ] **Large File**: Test with images > 10MB
- [ ] **Multiple Uploads**: Upload different images sequentially

#### Expected Behavior:
- ✅ Valid images should display in the main panel
- ❌ Invalid files should show error message
- 📏 Images should auto-resize if too large

### 3. **Domain Adaptation (CycleGAN) Testing**
#### Test Steps:
1. Upload an image
2. Select domain: "Synthetic to SEM" or "SEM to Synthetic"
3. Click "Apply Domain Adaptation"
4. Verify side-by-side comparison

#### Expected Results:
- Original and adapted images display side-by-side
- Processing indicator shows during adaptation
- Download button appears for adapted image

### 4. **Hotspot Classification Testing**
#### Test Different Models:
- [ ] **ResNet18**: Select and run classification
- [ ] **Vision Transformer (ViT)**: Test advanced model
- [ ] **EfficientNet-B0**: Test efficient model

#### Test Different Thresholds:
- [ ] **Low Threshold (0.3)**: Should detect more hotspots
- [ ] **Medium Threshold (0.5)**: Balanced detection
- [ ] **High Threshold (0.7)**: Conservative detection

#### Expected Outputs:
- Prediction confidence score
- Hotspot/No Hotspot classification
- Processing time display

### 5. **Grad-CAM Visualization Testing**
#### Test Steps:
1. After classification, enable "Show Grad-CAM"
2. Try different target layers
3. Verify heatmap overlay

#### Expected Results:
- Heatmap overlay on original image
- Color legend for attention areas
- Layer selection affects visualization

### 6. **Batch Processing Testing**
#### Test Scenarios:
- [ ] **Multiple Files**: Upload 3-5 images
- [ ] **Mixed Formats**: Upload .jpg and .png files
- [ ] **Large Batch**: Test with 10+ images (if available)

#### Expected Behavior:
- Progress bar shows batch processing
- Results table shows all predictions
- Download all results option available

### 7. **Download Functionality Testing**
#### Test Downloads:
- [ ] **Adapted Image**: Download CycleGAN result
- [ ] **Grad-CAM Result**: Download visualization
- [ ] **Batch Results**: Download CSV with all predictions
- [ ] **Analysis Report**: Download detailed PDF report

### 8. **Error Handling Testing**
#### Error Scenarios:
- [ ] **No Image Uploaded**: Try classification without image
- [ ] **Network Issues**: Simulate connection problems
- [ ] **Corrupted Image**: Upload damaged image file
- [ ] **Memory Issues**: Upload extremely large image

#### Expected Error Messages:
- Clear, user-friendly error descriptions
- Suggestions for resolution
- App remains stable after errors

### 9. **Performance Testing**
#### Metrics to Check:
- [ ] **Load Time**: App startup < 10 seconds
- [ ] **Image Processing**: Each operation < 30 seconds
- [ ] **Memory Usage**: Monitor in browser dev tools
- [ ] **CPU Usage**: Check system resources

### 10. **Advanced Features Testing**
#### Model Information:
- [ ] **Architecture Details**: Expand model info sections
- [ ] **Parameter Counts**: Verify model specifications
- [ ] **Training Details**: Check mock training information

#### Settings & Configuration:
- [ ] **Processing Settings**: Adjust image preprocessing
- [ ] **Visualization Options**: Customize color schemes
- [ ] **Export Settings**: Configure output formats

## 🎯 Sample Test Images

### Creating Test Images:
1. **Small Test Image** (for quick testing)
2. **Large Test Image** (for performance testing)
3. **Different Formats** (.jpg, .png, .jpeg)
4. **Edge Cases** (very small, very large, unusual aspect ratios)

## 📊 Test Results Documentation

### Success Criteria:
- ✅ All core features work without crashes
- ✅ Error messages are helpful and clear
- ✅ Performance is acceptable for demo purposes
- ✅ UI is responsive and professional
- ✅ Downloads work correctly

### Common Issues & Solutions:
1. **Slow Performance**: Expected with mock implementations
2. **Memory Warnings**: Normal for image processing
3. **Layout Issues**: Refresh page or adjust window size
4. **Upload Failures**: Check file size and format

## 🚀 Advanced Testing

### For Developers:
```bash
# Check logs
streamlit run app_working.py --logger.level debug

# Test with different ports
streamlit run app_working.py --server.port 8502

# Performance profiling
python -m cProfile -o profile.stats app_working.py
```

### Browser Developer Tools:
- **Console**: Check for JavaScript errors
- **Network**: Monitor file upload/download
- **Performance**: Check rendering times
- **Memory**: Monitor usage during processing

## 📋 Test Report Template

### Test Session: [Date/Time]
- **Tester**: [Name]
- **Browser**: [Chrome/Firefox/Safari + Version]
- **OS**: [Windows/Mac/Linux]

### Results Summary:
- **Passed**: [ ] / [ ] tests
- **Failed**: [ ] tests
- **Critical Issues**: [None/List]
- **Minor Issues**: [None/List]

### Recommendations:
- [ ] Ready for production
- [ ] Needs minor fixes
- [ ] Requires major updates

---

## 🎉 Quick Start Testing

**5-Minute Test:**
1. Upload any image file
2. Try domain adaptation
3. Run hotspot classification
4. Enable Grad-CAM visualization
5. Download one result

**15-Minute Full Test:**
1. Test all models and thresholds
2. Try batch processing
3. Test error handling
4. Check all download options
5. Test theme toggle and UI

Happy Testing! 🧪✨
