# AgriSprayAI - AI-Pesticide-Spraying-Optimizer

A production-grade, multimodal AI system that converts agricultural images and farmer notes into explainable per-plant pesticide prescriptions and executable UAV/ground nozzle schedules.

## 🎯 Overview

AgriSprayAI combines computer vision, natural language processing, and optimization algorithms to:
- Detect and segment pest/disease instances in agricultural images
- Extract severity levels (0-3) for each instance
- Process farmer notes (text/voice) for additional context
- Generate minimal guaranteed pesticide doses via convex optimization
- Create executable UAV flight plans with MAVLink/SDK compatibility
- Provide human-in-the-loop approval with full explainability

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision AI     │    │   NLP/ASR       │    │   Optimization  │
│   (YOLOv8 +     │    │   (Whisper +    │    │   (CVXPY +      │
│   Segmentation) │    │   GPT-3)        │    │   MILP)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Multimodal     │
                    │  Fusion         │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Planner &      │
                    │  MAVLink        │
                    └─────────────────┘
```

## 📁 Project Structure

```
project_root/
├── data/
│   ├── raw/                    # Raw datasets
│   ├── annotated/              # COCO format annotations
│   ├── masks/                  # Segmentation masks
│   └── field/                  # Field trial data
├── code/
│   ├── data_prep/              # Data preparation scripts
│   ├── vision/                 # Computer vision models
│   ├── nlp/                    # NLP and ASR pipelines
│   ├── fusion/                 # Multimodal fusion
│   ├── action_engine/          # Dose optimization
│   ├── planner/                # Flight planning
│   └── deploy/                 # Deployment scripts
├── models/                     # Trained model artifacts
├── notebooks/                  # Jupyter notebooks
├── deploy/                     # Docker and deployment configs
├── docs/                       # Documentation
└── logs/                       # System logs
```

## 🚀 Quick Start

### 🆕 **COMPLETELY NEW TO PROGRAMMING?**
**👉 [ULTIMATE BEGINNER GUIDE](ULTIMATE_BEGINNER_GUIDE.md) 👈**

**Perfect for complete beginners with zero programming experience!**

### 📚 **Available Guides (Choose Your Level):**

#### **🥇 For Complete Beginners:**
- **[🌟 ULTIMATE BEGINNER GUIDE](ULTIMATE_BEGINNER_GUIDE.md)** - Zero programming knowledge needed
- **[✅ Beginner Checklist](BEGINNER_CHECKLIST.md)** - Step-by-step checklist
- **[🚀 WORKING STARTUP GUIDE](WORKING_STARTUP_GUIDE.md)** - Proven working methods
- **[🎉 FINAL WORKING SOLUTION](FINAL_WORKING_SOLUTION.md)** - **RECOMMENDED** - Just double-click to start!

#### **🥈 For Those with Some Experience:**
- **[🚀 Quick Start Guide](QUICK_START_GUIDE.md)** - Get running in 5 minutes
- **[📖 Complete Setup Guide](BEGINNER_SETUP_GUIDE.md)** - Full step-by-step setup

#### **🥉 For Troubleshooting:**
- **[🆘 Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)** - Fix common issues
- **[✅ Setup Complete Summary](COMPLETE_SETUP_SUMMARY.md)** - What's been accomplished

### **Quick Start Options:**

#### **Option 1: Super Easy (Recommended for Beginners)**
```bash
# Just double-click this file on Windows:
start_project.bat

# Or run this command:
python start_project.py
```

#### **Option 2: Docker (One Command)**
```bash
# Make sure Docker Desktop is running, then:
docker-compose up --build
```

#### **Option 3: Manual Setup**
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt

# 2. Set up environment variables
copy env.example .env
# Edit .env with your API keys

# 3. Start the system
python code/api/server.py
```

### Prerequisites
- **Docker Desktop** (recommended) OR **Python 3.9-3.11** (NOT 3.13)
- **Git** for downloading the project
- **API Keys**: Google Gemini and Kaggle (see beginner guide)

## 🔧 Core Components

### 1. Vision Pipeline
- **Detector**: YOLOv8 (Ultralytics) with custom severity head
- **Segmentation**: Mask2Former or YOLOv8-seg for instance masks
- **Loss**: Multi-task loss combining detection, segmentation, and severity

### 2. NLP Pipeline
- **ASR**: Whisper (offline for privacy)
- **Text Embedding**: sentence-transformers (all-MiniLM-L6-v2)
- **GPT-3**: Structured symptom extraction and human-readable rationales

### 3. Multimodal Fusion
- **Concatenation**: Simple MLP fusion for edge deployment
- **Cross-attention**: Advanced attention mechanism for cloud deployment

### 4. Dose Optimization
- **Convex Solver**: CVXPY with OSQP/ECOS solvers
- **MILP Fallback**: OR-Tools for discrete actuator constraints
- **Safety**: Operator approval required for low confidence or regulatory exceedance

### 5. Flight Planning
- **Clustering**: DBSCAN for plant grouping
- **Routing**: TSP-like optimization for efficient paths
- **Output**: MAVLink missions and DJI SDK compatibility

## 🛡️ Safety & Compliance

- **Operator Approval**: Required for confidence < 0.80 or regulatory exceedance
- **Immutable Logging**: All decisions logged with full traceability
- **Privacy**: Farmer data requires explicit consent
- **Rate Limiting**: GPT-3 usage controlled to prevent abuse

## 📊 Performance Targets

- **Detection mAP@0.5**: ≥ 0.65
- **Edge Inference**: ≤ 0.8s per image
- **Pesticide Reduction**: ≥ 30% vs uniform spray
- **Operator Approval Rate**: ≤ 25% for automated plans

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Integration tests
pytest tests/integration/

# SITL simulation
python code/planner/sitl_simulator.py
```

## 📚 Documentation

- [Model Cards](docs/model_cards/)
- [API Documentation](docs/api/)
- [Operator Manual](docs/operator_manual.md)
- [Safety Checklist](docs/safety_checklist.md)
- [Developer Guide](docs/developer_guide.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is designed for research and pilot deployment. Always consult with agricultural experts and follow local regulations before field deployment. The system requires human oversight and approval for all pesticide applications.