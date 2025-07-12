#!/bin/bash

# AI Lithography Hotspot Detection - Launcher Script
# A comprehensive setup and launch script for all platforms

set -e  # Exit on any error

echo "🔬 AI Lithography Hotspot Detection System"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_blue() {
    echo -e "${BLUE}$1${NC}"
}

# Check if we're on Windows (Git Bash/WSL) or Unix-like system
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
    PYTHON_CMD="python"
    VENV_ACTIVATE=".venv/Scripts/activate"
    PIP_CMD="pip"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    PYTHON_CMD="python3"
    VENV_ACTIVATE=".venv/bin/activate"
    PIP_CMD="pip3"
else
    PLATFORM="linux"
    PYTHON_CMD="python3"
    VENV_ACTIVATE=".venv/bin/activate"
    PIP_CMD="pip3"
fi

print_blue "Detected platform: $PLATFORM"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python
if command_exists "$PYTHON_CMD"; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_status "Python found: $PYTHON_VERSION"
else
    print_error "Python not found. Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Check Git
if command_exists "git"; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    print_status "Git found: $GIT_VERSION"
else
    print_warning "Git not found. You may need it for updates and contributions."
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    print_status "Virtual environment created successfully"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_ACTIVATE"
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip > /dev/null 2>&1

# Check if requirements are installed
print_status "Checking dependencies..."
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please ensure you're in the project directory."
    exit 1
fi

# Install/update requirements
print_status "Installing/updating dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    exit 1
fi

print_status "Dependencies installed successfully"
echo ""

# Function to show menu
show_menu() {
    echo ""
    print_blue "🚀 Choose your option:"
    echo "1) Launch Advanced Application (Full AI features)"
    echo "2) Launch Basic Application (Lightweight version)"
    echo "3) View Setup Guide (Complete tutorial)"
    echo "4) Run Tests"
    echo "5) Update Project"
    echo "6) Check System Status"
    echo "7) Exit"
    echo ""
}

# Function to launch advanced app
launch_advanced() {
    print_status "Launching Advanced AI Application..."
    print_blue "🌐 Application will open in your default browser"
    print_blue "📍 URL: http://localhost:8501"
    echo ""
    streamlit run app_advanced.py
}

# Function to launch basic app
launch_basic() {
    print_status "Launching Basic Application..."
    print_blue "🌐 Application will open in your default browser"
    print_blue "📍 URL: http://localhost:8501"
    echo ""
    streamlit run app.py
}

# Function to show setup guide
show_setup_guide() {
    print_status "Opening Complete Setup Guide..."
    if [ -f "COMPLETE_SETUP_GUIDE.md" ]; then
        if command_exists "code"; then
            code COMPLETE_SETUP_GUIDE.md
            print_status "Setup guide opened in VS Code"
        elif command_exists "notepad" && [ "$PLATFORM" == "windows" ]; then
            notepad COMPLETE_SETUP_GUIDE.md
            print_status "Setup guide opened in Notepad"
        else
            print_blue "📚 Please open COMPLETE_SETUP_GUIDE.md in your preferred text editor"
            print_blue "📁 File location: $(pwd)/COMPLETE_SETUP_GUIDE.md"
        fi
    else
        print_error "Setup guide not found. Please ensure COMPLETE_SETUP_GUIDE.md exists."
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    if [ -f "test_basic.py" ]; then
        python test_basic.py
    else
        print_warning "No test files found"
    fi
}

# Function to update project
update_project() {
    print_status "Updating project..."
    if command_exists "git"; then
        git pull origin main
        pip install -r requirements.txt
        print_status "Project updated successfully"
    else
        print_error "Git not available. Please update manually."
    fi
}

# Function to check system status
check_status() {
    print_status "System Status Check"
    echo "================================"
    
    # Python version
    echo "Python: $($PYTHON_CMD --version)"
    
    # Virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Virtual Environment: Active ($(basename $VIRTUAL_ENV))"
    else
        echo "Virtual Environment: Not active"
    fi
    
    # GPU support
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch: Not installed"
    
    # Streamlit
    streamlit --version 2>/dev/null || echo "Streamlit: Not installed"
    
    # Memory
    python -c "import psutil; print(f'Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB')" 2>/dev/null || echo "Memory info: Not available"
    
    echo "================================"
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            launch_advanced
            ;;
        2)
            launch_basic
            ;;
        3)
            show_setup_guide
            ;;
        4)
            run_tests
            ;;
        5)
            update_project
            ;;
        6)
            check_status
            ;;
        7)
            print_status "Goodbye! Thank you for using AI Lithography Hotspot Detection"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please enter 1-7."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
