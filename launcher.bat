@echo off
title Lithography Hotspot Detection - Launcher
color 0B

echo ================================================================
echo    🔬 Lithography Hotspot Detection Application Launcher
echo ================================================================
echo.
echo Select the application mode:
echo.
echo [1] 🚀 Advanced App (Full AI Models, GPU Acceleration)
echo [2] 📱 Basic App (Mock Models, Lightweight)
echo [3] ❌ Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto advanced
if "%choice%"=="2" goto basic
if "%choice%"=="3" goto exit
echo Invalid choice. Please try again.
pause
goto start

:advanced
echo.
echo ================================================================
echo Starting Advanced Lithography Hotspot Detection App...
echo ================================================================
echo.
echo Features:
echo • Real PyTorch AI Models (ResNet18, ViT, EfficientNet)
echo • Advanced CycleGAN Domain Adaptation
echo • GPU Acceleration Support
echo • Comprehensive Grad-CAM Visualizations
echo • Advanced Image Processing Algorithms
echo • Analytics Dashboard
echo • Model Management Interface
echo.
cd /d "C:\Users\Abhinav\Desktop\Mainprojects"
call ".venv\Scripts\activate.bat"
echo Virtual environment activated
echo.
echo Starting Advanced App on http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
python -m streamlit run app_advanced.py --server.port=8501
goto end

:basic
echo.
echo ================================================================
echo Starting Basic Lithography Hotspot Detection App...
echo ================================================================
echo.
echo Features:
echo • Mock AI Models (Lightweight)
echo • Basic Image Processing
echo • Simple Classification
echo • Standard Visualizations
echo • About Page with Resources
echo.
cd /d "C:\Users\Abhinav\Desktop\Mainprojects"
call ".venv\Scripts\activate.bat"
echo Virtual environment activated
echo.
echo Starting Basic App on http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
python -m streamlit run app_working.py --server.port=8501
goto end

:exit
echo.
echo Goodbye! 👋
timeout /t 2 >nul
exit

:end
echo.
echo ================================================================
echo Application stopped. Thank you for using the app! 🙏
echo ================================================================
pause
