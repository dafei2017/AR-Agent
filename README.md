# AR-Agent: Medical Multimodal Augmented Reality Agent

## Overview

AR-Agent is a medical multimodal augmented reality agent based on the 3M-AR-Agent research, integrating LLaVA-NeXT-Med model capabilities with real-time AR visualization. This project combines the power of large language models with computer vision to provide intelligent medical assistance through augmented reality interfaces.

## Features

- **Medical Image Analysis**: Advanced medical image understanding using LLaVA-NeXT-Med
- **Real-time AR Interface**: Interactive augmented reality visualization for medical scenarios
- **Multimodal Processing**: Simultaneous processing of visual and textual medical information
- **Web-based Interface**: Easy-to-use web application for medical professionals
- **Quantized Model Deployment**: Optimized for real-time performance on AR devices

## Architecture

The AR-Agent system consists of three main components:

1. **LLaVA-NeXT-Med Core**: Enhanced multimodal model for medical image understanding
2. **AR Interface Module**: Real-time augmented reality visualization
3. **Web Application**: User-friendly interface for medical professionals

## Key Components

### 1. Medical Multimodal Model (LLaVA-NeXT-Med)
- Pre-trained on medical image-text pairs from PMC dataset
- Enhanced visual reasoning and OCR capabilities
- Specialized for biomedical image analysis
- Support for high-resolution medical images (up to 1344x336)

### 2. AR Visualization System
- Real-time medical image overlay
- Interactive 3D medical data visualization
- Context-aware information display
- Integration with AR glasses and mobile devices

### 3. Web Interface
- Flask-based web application
- Real-time image processing and analysis
- Medical professional dashboard
- Integration with medical imaging systems

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- AR-compatible device (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/dafei2017/AR-Agent.git
cd AR-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional medical packages
pip install -r requirements-medical.txt
```

## Usage

### Web Application

```bash
# Start the web server
python app.py

# Access the application at http://localhost:5000
```

### AR Interface

```bash
# Start AR interface (requires AR device)
python ar_interface.py
```

### Medical Image Analysis

```python
from ar_agent import MedicalAnalyzer

# Initialize the analyzer
analyzer = MedicalAnalyzer()

# Analyze medical image
result = analyzer.analyze_image("path/to/medical_image.jpg")
print(result.description)
print(result.findings)
```

## Model Training

### Pre-training

```bash
# Pre-train on medical datasets
bash scripts/pretrain_medical.sh
```

### Fine-tuning

```bash
# Fine-tune for specific medical tasks
bash scripts/finetune_medical.sh
```

## Evaluation

The model has been evaluated on multiple medical VQA datasets:

- **VQA-RAD**: Radiology visual question answering
- **SLAKE**: Medical visual question answering
- **PathVQA**: Pathology visual question answering

## Performance

- **Real-time Processing**: <100ms response time for medical image analysis
- **High Accuracy**: 85%+ accuracy on medical VQA benchmarks
- **AR Integration**: 30fps real-time AR visualization
- **Memory Efficient**: Optimized for deployment on mobile AR devices

## Applications

1. **Surgical Assistance**: Real-time guidance during medical procedures
2. **Medical Education**: Interactive learning for medical students
3. **Diagnostic Support**: AI-assisted medical image interpretation
4. **Patient Care**: Enhanced patient-doctor communication
5. **Telemedicine**: Remote medical consultation with AR support

## Research Background

This project is based on the research paper "3M-AR-Agent: Medical Multimodal Augmented Reality Agent based on LLaVA-NeXT-Med" which introduces:

- Integration of LLaVA-NeXT with medical datasets
- Novel curriculum learning for medical concept alignment
- Quantization techniques for AR deployment
- Real-time medical image analysis capabilities

## Contributing

We welcome contributions to improve AR-Agent. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use AR-Agent in your research, please cite:

```bibtex
@article{guo2024ar_agent,
  title={3M-AR-Agent: Medical Multimodal Augmented Reality Agent based on LLaVA-NeXT-Med},
  author={Guo, Y},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

- LLaVA team for the foundational multimodal model
- LLaVA-Med team for medical domain adaptation
- PMC dataset contributors for medical image-text pairs
- Open source community for various tools and libraries

## Contact

- **Yunfei Guo**: guoyunfei@tme.com.cn
- **Project Repository**: https://github.com/dafei2017/AR-Agent

---

**Note**: This project is for research and educational purposes. Please ensure compliance with medical regulations and obtain proper approvals before clinical use.
