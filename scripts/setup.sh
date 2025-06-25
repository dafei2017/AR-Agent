#!/bin/bash
# AR-Agent Setup Script
# Initializes the environment for the Medical Multimodal Augmented Reality Agent

set -e

# Define colors for output
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Print banner
echo -e "${GREEN}"
echo "========================================================="
echo "   AR-Agent Setup - Medical Multimodal AR Agent"
echo "========================================================="
echo -e "${NC}"

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 8 ]; then
    echo -e "${RED}Error: Python 3.8 or higher is required. Found Python $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Using Python $PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies if requested
if [ "$1" == "--dev" ]; then
    echo -e "${YELLOW}Installing development dependencies...${NC}"
    pip install -e ".[dev]"
    
    # Setup pre-commit hooks
    echo -e "${YELLOW}Setting up pre-commit hooks...${NC}"
    pre-commit install
else
    # Install the package
    echo -e "${YELLOW}Installing AR-Agent package...${NC}"
    pip install -e .
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p data/images data/results data/models data/cache logs uploads

# Check for GPU availability
echo -e "${YELLOW}Checking for GPU availability...${NC}"
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}GPU is available!${NC}"
    python -c "import torch; print('CUDA Device:', torch.cuda.get_device_name(0))"
else
    echo -e "${YELLOW}No GPU detected. AR-Agent will run on CPU mode.${NC}"
fi

# Download model if requested
if [ "$1" == "--download-model" ] || [ "$2" == "--download-model" ]; then
    echo -e "${YELLOW}Downloading LLaVA-NeXT model...${NC}"
    mkdir -p models
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, LlavaNextProcessor, LlavaNextForConditionalGeneration; LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf'); LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')"
    echo -e "${GREEN}Model downloaded successfully!${NC}"
fi

# Setup complete
echo -e "${GREEN}"
echo "========================================================="
echo "   AR-Agent Setup Complete!"
echo "========================================================="
echo -e "${NC}"
echo "To activate the environment, run:"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "To start the application, run:"
echo -e "${YELLOW}python app.py${NC}"
echo ""
echo "For development with hot reload:"
echo -e "${YELLOW}FLASK_ENV=development FLASK_DEBUG=1 python app.py${NC}"
echo ""
echo "To run with Docker:"
echo -e "${YELLOW}docker-compose up${NC}"
echo ""
echo "Documentation available at: https://github.com/dafei2017/AR-Agent"
echo -e "${GREEN}Happy AR-Agent development!${NC}"