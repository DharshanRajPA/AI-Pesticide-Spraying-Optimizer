# AI Pesticide Spraying Optimizer - Installation Guide

## Overview
This guide provides step-by-step instructions for setting up the AI Pesticide Spraying Optimizer project on Windows with Python 3.13.

## Prerequisites
- Python 3.13
- Windows 10/11
- Git (for cloning detectron2)

## Installation Steps

### 1. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Main Requirements
```bash
pip install -r requirements.txt
```

### 3. Install Special Packages

#### Detectron2
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

#### MMCV (Optional - for advanced computer vision)
If you need MMCV, install it separately:
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### 4. Verify Installation
```bash
python -c "import torch; import ultralytics; import detectron2; import cv2; print('All key packages imported successfully!')"
```

## Platform-Specific Notes

### Windows Compatibility
- **TensorRT**: Not available on Windows (commented out in requirements.txt)
- **TFLite Runtime**: Not available on Windows (commented out in requirements.txt)
- **MMCV**: May require special installation steps on Windows

### Python 3.13 Compatibility
- **Ultralytics**: Updated to version 8.3.0+ for Python 3.13 support
- **Detectron2**: Installed from source to ensure compatibility

## Troubleshooting

### Common Issues

1. **Detectron2 Installation Fails**
   - Solution: Install from source using the git command above
   - Alternative: Use pre-built wheels if available for your CUDA version

2. **MMCV Build Errors**
   - Solution: Install pre-built wheels from OpenMMLab
   - Check CUDA version compatibility

3. **TensorRT/TFLite Runtime Errors**
   - These are platform-specific and not needed for Windows development
   - They're commented out in requirements.txt

### CUDA Support
If you have CUDA installed and want GPU acceleration:
- Ensure your CUDA version is compatible with PyTorch
- Install CUDA-enabled versions of packages when available

## Development Setup

### Code Quality Tools
The following tools are installed for development:
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `pytest` - Testing

### Pre-commit Hooks
```bash
pre-commit install
```

## Next Steps
1. Configure your environment variables in `.env`
2. Set up your data directories in the `data/` folder
3. Run tests to verify everything works: `pytest tests/`

## Support
If you encounter issues:
1. Check the troubleshooting section above
2. Verify your Python and CUDA versions
3. Check package compatibility with Python 3.13
