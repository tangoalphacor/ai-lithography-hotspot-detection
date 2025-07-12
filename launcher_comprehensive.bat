@echo off
setlocal enabledelayedexpansion
title AI Lithography Hotspot Detection - Smart Launcher
color 0B

:: Enable ANSI escape sequences for colored output
for /f "tokens=2 delims=[]" %%I in ('ver') do set winver=%%I
if not "!winver:10.0=!" == "!winver!" (
    :: Windows 10/11 - enable ANSI colors
    reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1
)

:: Color codes
set "GREEN=[92m"
set "BLUE=[94m"
set "YELLOW=[93m"
set "RED=[91m"
set "RESET=[0m"

:start
cls
echo %GREEN%================================================================%RESET%
echo    🔬 AI Lithography Hotspot Detection System
echo %GREEN%================================================================%RESET%
echo.
echo %BLUE%A professional-grade AI system for semiconductor manufacturing%RESET%
echo.

:: Check if virtual environment exists
if not exist ".venv" (
    echo %YELLOW%[WARNING]%RESET% Virtual environment not found. Creating one...
    python -m venv .venv
    if errorlevel 1 (
        echo %RED%[ERROR]%RESET% Failed to create virtual environment
        echo Please ensure Python 3.8+ is installed and in your PATH
        pause
        exit /b 1
    )
    echo %GREEN%[SUCCESS]%RESET% Virtual environment created
    echo.
)

:: Activate virtual environment
echo %GREEN%[INFO]%RESET% Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo %RED%[ERROR]%RESET% Failed to activate virtual environment
    pause
    exit /b 1
)

:: Check if requirements are installed
echo %GREEN%[INFO]%RESET% Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[INFO]%RESET% Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo %RED%[ERROR]%RESET% Failed to install dependencies
        pause
        exit /b 1
    )
)

:menu
cls
echo %GREEN%================================================================%RESET%
echo    🔬 AI Lithography Hotspot Detection System
echo %GREEN%================================================================%RESET%
echo.
echo %BLUE%🚀 Main Applications:%RESET%
echo [1] 🤖 Advanced Application (Full AI models, GPU acceleration)
echo [2] 📱 Basic Application (Lightweight, demo version)
echo.
echo %BLUE%📚 Help ^& Resources:%RESET%
echo [3] 📖 Complete Setup Guide (Step-by-step tutorial)
echo [4] 🎨 Test Image Generator (Create sample images)
echo [5] 📊 View Documentation (README, guides, examples)
echo.
echo %BLUE%🛠️ Maintenance:%RESET%
echo [6] 🔄 Update Project (Git pull latest changes)
echo [7] 🔍 System Status (Check installation and performance)
echo [8] 🧹 Clean Environment (Reset virtual environment)
echo.
echo [9] ❌ Exit
echo.
set /p choice="%BLUE%Enter your choice (1-9): %RESET%"

if "%choice%"=="1" goto advanced
if "%choice%"=="2" goto basic
if "%choice%"=="3" goto setup_guide
if "%choice%"=="4" goto test_generator
if "%choice%"=="5" goto documentation
if "%choice%"=="6" goto update
if "%choice%"=="7" goto status
if "%choice%"=="8" goto clean
if "%choice%"=="9" goto exit
echo %RED%[ERROR]%RESET% Invalid choice. Please enter 1-9.
timeout /t 2 >nul
goto menu

:advanced
cls
echo %GREEN%================================================================%RESET%
echo    🚀 Launching Advanced AI Application
echo %GREEN%================================================================%RESET%
echo.
echo %BLUE%Features enabled:%RESET%
echo • 🧠 Real PyTorch AI Models (ResNet18, ViT, EfficientNet)
echo • 🔄 Advanced CycleGAN Domain Adaptation
echo • ⚡ GPU Acceleration Support
echo • 🔍 Comprehensive Grad-CAM Visualizations
echo • 📊 Analytics Dashboard and Model Management
echo • 🎨 Advanced Test Image Generator
echo • 📈 Performance Analytics and Reporting
echo.
echo %GREEN%[INFO]%RESET% Starting advanced application...
echo %GREEN%[INFO]%RESET% URL: http://localhost:8501
echo %GREEN%[INFO]%RESET% Press Ctrl+C to stop the server
echo.
streamlit run app_advanced.py --server.port=8501
goto continue

:basic
cls
echo %GREEN%================================================================%RESET%
echo    📱 Launching Basic Application
echo %GREEN%================================================================%RESET%
echo.
echo %BLUE%Features enabled:%RESET%
echo • 🎯 Mock AI Models (Lightweight demonstration)
echo • 📷 Basic Image Processing
echo • 🔍 Simple Classification Interface
echo • 📊 Standard Visualizations
echo • 📋 About Page with Project Information
echo.
echo %GREEN%[INFO]%RESET% Starting basic application...
echo %GREEN%[INFO]%RESET% URL: http://localhost:8501
echo %GREEN%[INFO]%RESET% Press Ctrl+C to stop the server
echo.
if exist "app.py" (
    streamlit run app.py --server.port=8501
) else (
    echo %YELLOW%[WARNING]%RESET% Basic app not found, launching advanced app instead
    streamlit run app_advanced.py --server.port=8501
)
goto continue

:setup_guide
cls
echo %GREEN%================================================================%RESET%
echo    📖 Opening Complete Setup Guide
echo %GREEN%================================================================%RESET%
echo.
echo %GREEN%[INFO]%RESET% The Complete Setup Guide contains:
echo • 🔬 Introduction to Lithography and Hotspots
echo • 🤖 AI Technologies Explained
echo • 💻 Step-by-step Installation Instructions
echo • 🎯 How to Use the Application
echo • 🐛 Troubleshooting Common Issues
echo • 💾 Git Basics for Beginners
echo • 🛠️ Customization and Extension Guide
echo.

if exist "COMPLETE_SETUP_GUIDE.md" (
    echo %GREEN%[INFO]%RESET% Opening setup guide...
    
    :: Try different editors in order of preference
    where code >nul 2>&1
    if !errorlevel! equ 0 (
        code "COMPLETE_SETUP_GUIDE.md"
        echo %GREEN%[SUCCESS]%RESET% Setup guide opened in VS Code
    ) else (
        where notepad++ >nul 2>&1
        if !errorlevel! equ 0 (
            "notepad++" "COMPLETE_SETUP_GUIDE.md"
            echo %GREEN%[SUCCESS]%RESET% Setup guide opened in Notepad++
        ) else (
            start "" "COMPLETE_SETUP_GUIDE.md"
            echo %GREEN%[SUCCESS]%RESET% Setup guide opened in default markdown viewer
        )
    )
) else (
    echo %RED%[ERROR]%RESET% Setup guide not found: COMPLETE_SETUP_GUIDE.md
    echo Please ensure you have the complete project files
)
goto continue

:test_generator
cls
echo %GREEN%================================================================%RESET%
echo    🎨 Test Image Generator
echo %GREEN%================================================================%RESET%
echo.
echo %GREEN%[INFO]%RESET% Launching application with Test Image Generator page...
echo %GREEN%[INFO]%RESET% Navigate to "Test Image Generator" once the app loads
echo.
streamlit run app_advanced.py --server.port=8501
goto continue

:documentation
cls
echo %GREEN%================================================================%RESET%
echo    📚 Project Documentation
echo %GREEN%================================================================%RESET%
echo.
echo %BLUE%Available documentation:%RESET%
echo.

if exist "README.md" (
    echo ✅ README.md - Project overview and quick start
) else (
    echo ❌ README.md - Not found
)

if exist "COMPLETE_SETUP_GUIDE.md" (
    echo ✅ COMPLETE_SETUP_GUIDE.md - Comprehensive setup tutorial
) else (
    echo ❌ COMPLETE_SETUP_GUIDE.md - Not found
)

if exist "DEPLOYMENT_GUIDE.md" (
    echo ✅ DEPLOYMENT_GUIDE.md - Cloud deployment instructions
) else (
    echo ❌ DEPLOYMENT_GUIDE.md - Not found
)

if exist "TEST_IMAGE_GENERATOR_DOCS.md" (
    echo ✅ TEST_IMAGE_GENERATOR_DOCS.md - Test image generator guide
) else (
    echo ❌ TEST_IMAGE_GENERATOR_DOCS.md - Not found
)

echo.
echo %BLUE%Opening project folder in file explorer...%RESET%
start "" "%cd%"
goto continue

:update
cls
echo %GREEN%================================================================%RESET%
echo    🔄 Updating Project
echo %GREEN%================================================================%RESET%
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%RESET% Git not found. Please install Git to enable updates.
    echo Download from: https://git-scm.com/download/win
    goto continue
)

echo %GREEN%[INFO]%RESET% Checking for updates...
git fetch origin main
git status --porcelain | find /c /v "" >nul
if not errorlevel 1 (
    echo %YELLOW%[WARNING]%RESET% You have local changes. Stashing them...
    git stash push -m "Auto-stash before update"
)

echo %GREEN%[INFO]%RESET% Pulling latest changes...
git pull origin main
if errorlevel 1 (
    echo %RED%[ERROR]%RESET% Failed to update. Check your internet connection.
    goto continue
)

echo %GREEN%[INFO]%RESET% Updating dependencies...
pip install -r requirements.txt --upgrade
echo %GREEN%[SUCCESS]%RESET% Project updated successfully!
goto continue

:status
cls
echo %GREEN%================================================================%RESET%
echo    🔍 System Status Check
echo %GREEN%================================================================%RESET%
echo.

echo %BLUE%Python Environment:%RESET%
python --version 2>nul
if errorlevel 1 (
    echo %RED%❌ Python not found%RESET%
) else (
    echo %GREEN%✅ Python installed%RESET%
)

echo.
echo %BLUE%Virtual Environment:%RESET%
if defined VIRTUAL_ENV (
    echo %GREEN%✅ Virtual environment active%RESET%
    echo    Path: %VIRTUAL_ENV%
) else (
    echo %YELLOW%⚠️ Virtual environment not active%RESET%
)

echo.
echo %BLUE%Key Dependencies:%RESET%
for %%p in (streamlit torch torchvision pillow numpy pandas) do (
    pip show %%p >nul 2>&1
    if errorlevel 1 (
        echo %RED%❌ %%p - Not installed%RESET%
    ) else (
        for /f "tokens=2" %%v in ('pip show %%p 2^>nul ^| findstr "Version:"') do (
            echo %GREEN%✅ %%p - %%v%RESET%
        )
    )
)

echo.
echo %BLUE%System Resources:%RESET%
python -c "import psutil; print(f'Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB')" 2>nul
if errorlevel 1 (
    echo %YELLOW%⚠️ Could not check memory usage%RESET%
)

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo %YELLOW%⚠️ Could not check GPU status%RESET%
)

echo.
echo %BLUE%Project Files:%RESET%
if exist "app_advanced.py" (
    echo %GREEN%✅ Advanced application%RESET%
) else (
    echo %RED%❌ Advanced application missing%RESET%
)

if exist "requirements.txt" (
    echo %GREEN%✅ Requirements file%RESET%
) else (
    echo %RED%❌ Requirements file missing%RESET%
)

if exist "COMPLETE_SETUP_GUIDE.md" (
    echo %GREEN%✅ Setup guide%RESET%
) else (
    echo %RED%❌ Setup guide missing%RESET%
)

goto continue

:clean
cls
echo %GREEN%================================================================%RESET%
echo    🧹 Clean Environment
echo %GREEN%================================================================%RESET%
echo.
echo %YELLOW%[WARNING]%RESET% This will delete the virtual environment and reinstall it.
echo All installed packages will need to be downloaded again.
echo.
set /p confirm="Are you sure? (y/N): "
if /i not "%confirm%"=="y" goto continue

echo %GREEN%[INFO]%RESET% Deactivating virtual environment...
call deactivate 2>nul

echo %GREEN%[INFO]%RESET% Removing old virtual environment...
if exist ".venv" rmdir /s /q ".venv"

echo %GREEN%[INFO]%RESET% Creating new virtual environment...
python -m venv .venv
call ".venv\Scripts\activate.bat"

echo %GREEN%[INFO]%RESET% Installing dependencies...
pip install -r requirements.txt

echo %GREEN%[SUCCESS]%RESET% Environment cleaned and restored!
goto continue

:continue
echo.
echo %GREEN%================================================================%RESET%
echo.
set /p return="Press Enter to return to main menu..."
goto menu

:exit
cls
echo %GREEN%================================================================%RESET%
echo    👋 Thank You for Using AI Lithography Hotspot Detection
echo %GREEN%================================================================%RESET%
echo.
echo %BLUE%Built with ❤️ by Abhinav%RESET%
echo %BLUE%Made for the global community of developers and engineers%RESET%
echo.
echo %GREEN%Remember: Every expert was once a beginner. You've got this! 🌟%RESET%
echo.
timeout /t 3 >nul
exit /b 0

:error
echo %RED%[ERROR]%RESET% An unexpected error occurred.
pause
exit /b 1
