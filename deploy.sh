#!/bin/bash
# Simple deployment script for Streamlit Cloud

echo "🚀 Deploying AI Lithography Hotspot Detection App"
echo "=================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git branch -M main
fi

# Add all files
echo "📦 Adding files to Git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deploy AI Lithography Hotspot Detection App v$(date +%Y%m%d_%H%M%S)"

# Check for remote origin
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "⚠️  Please set up your GitHub repository and add remote origin:"
    echo "   git remote add origin https://github.com/yourusername/ai-lithography-hotspot-detection.git"
    echo "   git push -u origin main"
else
    echo "⬆️  Pushing to GitHub..."
    git push origin main
    echo "✅ Deployment complete!"
    echo "🌐 Visit https://share.streamlit.io to deploy your app"
fi
