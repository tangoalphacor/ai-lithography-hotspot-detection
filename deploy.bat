@echo off
REM Windows deployment script for AI Lithography Hotspot Detection App

echo 🚀 Deploying AI Lithography Hotspot Detection App
echo ==================================

REM Check if git is initialized
if not exist ".git" (
    echo 📁 Initializing Git repository...
    git init
    git branch -M main
)

REM Add all files
echo 📦 Adding files to Git...
git add .

REM Commit changes
echo 💾 Committing changes...
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
git commit -m "Deploy AI Lithography Hotspot Detection App v%mydate%_%mytime%"

REM Check for remote origin
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Please set up your GitHub repository and add remote origin:
    echo    git remote add origin https://github.com/yourusername/ai-lithography-hotspot-detection.git
    echo    git push -u origin main
) else (
    echo ⬆️  Pushing to GitHub...
    git push origin main
    echo ✅ Deployment complete!
    echo 🌐 Visit https://share.streamlit.io to deploy your app
)

pause
