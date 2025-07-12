# Quick Setup Instructions for GitHub Deployment

## 🚀 Quick Start (5 minutes)

### 1. Create GitHub Repository
```bash
# Option A: Using GitHub CLI (recommended)
gh repo create ai-lithography-hotspot-detection --public --description "AI-powered lithography hotspot detection using deep learning"

# Option B: Manual setup
# 1. Go to https://github.com/new
# 2. Repository name: ai-lithography-hotspot-detection
# 3. Description: AI-powered lithography hotspot detection using deep learning
# 4. Make it public
# 5. Create repository
```

### 2. Push to GitHub
```bash
# Initialize and push (Windows PowerShell)
git init
git branch -M main
git add .
git commit -m "Initial commit: AI Lithography Hotspot Detection App"
git remote add origin https://github.com/YOURUSERNAME/ai-lithography-hotspot-detection.git
git push -u origin main
```

### 3. Deploy to Streamlit Cloud
1. Visit https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub repository
4. Set main file path: `app_advanced.py`
5. Click "Deploy!"

### 4. Enable GitHub Pages
1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: gh-pages (auto-created by GitHub Actions)
4. Your site will be at: `https://YOURUSERNAME.github.io/ai-lithography-hotspot-detection`

## 📝 Replace YOURUSERNAME
Don't forget to replace `YOURUSERNAME` with your actual GitHub username in:
- Repository URL
- README.md links
- GitHub Pages URL

## 🎯 Expected Results
- **Source Code**: https://github.com/YOURUSERNAME/ai-lithography-hotspot-detection
- **Live App**: https://ai-lithography-hotspot-detection.streamlit.app
- **Demo Site**: https://YOURUSERNAME.github.io/ai-lithography-hotspot-detection

## 🆘 Need Help?
- Check the full `GITHUB_DEPLOYMENT_GUIDE.md` for detailed instructions
- GitHub Issues: Create an issue in your repository
- Streamlit Support: https://docs.streamlit.io/streamlit-community-cloud
