# 🧪 Quick Testing Workflow

## Phase 1: Basic Functionality (5 minutes)

### ✅ Step 1: Verify App is Running
1. Open your browser to `http://localhost:8501`
2. Confirm the app loads with the professional dark theme
3. Check that all sidebar sections are visible:
   - File Upload
   - Domain Selection  
   - Model Configuration
   - Processing Options

### ✅ Step 2: Test Image Upload
1. Click "Choose an image file..." in the sidebar
2. Navigate to `test_images/` folder
3. Upload `small_test.jpg` (256x256 gradient image)
4. **Expected**: Image appears in main panel with original size info

### ✅ Step 3: Test Domain Adaptation
1. With image uploaded, select "Synthetic to SEM" 
2. Click "🔄 Apply Domain Adaptation"
3. **Expected**: 
   - Processing indicator shows briefly
   - Side-by-side comparison appears
   - Download button becomes available

### ✅ Step 4: Test Hotspot Classification
1. Keep default model (ResNet18) and threshold (0.5)
2. Click "🔍 Classify Hotspot"
3. **Expected**:
   - Prediction result shows (Hotspot/No Hotspot)
   - Confidence score displays
   - Processing time shown

### ✅ Step 5: Test Grad-CAM Visualization
1. Check "Show Grad-CAM Visualization"
2. Click "🎯 Generate Grad-CAM"
3. **Expected**:
   - Heatmap overlay appears on image
   - Color legend shows attention areas
   - Download option available

---

## Phase 2: Advanced Testing (10 minutes)

### ✅ Test Different Models
1. **Vision Transformer (ViT)**:
   - Select from dropdown
   - Run classification
   - Note different processing time
   
2. **EfficientNet-B0**:
   - Select from dropdown  
   - Run classification
   - Compare results

### ✅ Test Different Thresholds
1. Set threshold to **0.3** (sensitive)
   - Run classification
   - Note if more hotspots detected
   
2. Set threshold to **0.7** (conservative)
   - Run classification
   - Compare with previous results

### ✅ Test Different Image Sizes
1. Upload `medium_test.png` (512x512)
2. Upload `large_test.jpg` (1024x1024)
3. Upload `wide_test.png` (400x800)
4. Upload `tall_test.jpg` (600x300)

**Expected**: All images should process correctly with appropriate resizing

### ✅ Test Batch Processing
1. Click "Upload multiple files for batch processing"
2. Select 3-4 images from test_images folder
3. Click "🚀 Process Batch"
4. **Expected**:
   - Progress bar shows processing
   - Results table appears
   - Download CSV option available

---

## Phase 3: Error Handling & Edge Cases (5 minutes)

### ✅ Test Error Scenarios
1. **No Image**: Try classification without uploading
   - **Expected**: Clear error message
   
2. **Invalid File**: Try uploading a .txt file
   - **Expected**: File type error message
   
3. **Theme Toggle**: Switch between light/dark themes
   - **Expected**: Smooth transition, no layout breaks

### ✅ Test Downloads
1. Download adapted image (after domain adaptation)
2. Download Grad-CAM result (after visualization)
3. Download batch results CSV (after batch processing)

**Expected**: All downloads work correctly

---

## Phase 4: Performance & UI Testing (5 minutes)

### ✅ Performance Check
- Monitor processing times (should be fast with mock models)
- Check for memory warnings in browser
- Verify app remains responsive

### ✅ UI/UX Check
- Test responsive design (resize browser window)
- Verify all buttons and controls work
- Check that error messages are helpful
- Confirm professional appearance

---

## 🎯 Success Criteria

### ✅ Must Pass:
- [ ] App loads without errors
- [ ] Image upload works for valid files
- [ ] All three main features work:
  - [ ] Domain adaptation
  - [ ] Hotspot classification  
  - [ ] Grad-CAM visualization
- [ ] Error handling is graceful
- [ ] Downloads function correctly

### ✅ Should Pass:
- [ ] All models selectable and functional
- [ ] Batch processing works
- [ ] Theme toggle works
- [ ] Professional UI appearance
- [ ] Reasonable performance

### ✅ Nice to Have:
- [ ] Detailed model information accessible
- [ ] Smooth animations and transitions
- [ ] Comprehensive error messages
- [ ] Mobile-friendly responsive design

---

## 🐛 Common Issues & Solutions

### Issue: "Module not found" error
**Solution**: Ensure virtual environment is activated and all packages installed

### Issue: Images not displaying
**Solution**: Check file formats (.jpg, .png, .jpeg supported)

### Issue: Slow processing
**Solution**: Normal for demo - mock implementations simulate processing time

### Issue: Download not working
**Solution**: Check browser popup blockers and download settings

### Issue: Layout problems
**Solution**: Refresh page or adjust browser window size

---

## 🎉 Testing Complete!

If all phases pass, your Streamlit app is working correctly! 

### Next Steps:
1. **Deploy** (optional): Use Streamlit Cloud, Heroku, or Docker
2. **Enhance**: Add real ML models when ready
3. **Optimize**: Improve performance for production use
4. **Document**: Update README with deployment instructions

### Report Issues:
- Note any failed tests
- Screenshot any UI problems  
- Document error messages
- Check browser console for JavaScript errors

**Happy Testing! 🚀**
