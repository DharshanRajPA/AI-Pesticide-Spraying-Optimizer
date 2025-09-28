# AgriSprayAI Developer Guide

This guide provides comprehensive instructions for developers working on the AgriSprayAI system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Environment](#development-environment)
4. [Training Models](#training-models)
5. [API Development](#api-development)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Contributing](#contributing)

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker and Docker Compose
- CUDA 11.8+ (for GPU training)
- Git

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd AI-Pesticide-Spraying-Optimizer
```

2. **Set up Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up Node.js environment:**
```bash
cd ui
npm install
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Project Structure

```
project_root/
├── code/                    # Source code
│   ├── data_prep/          # Data preparation scripts
│   ├── vision/             # Computer vision models
│   ├── nlp/                # NLP and ASR pipelines
│   ├── fusion/             # Multimodal fusion
│   ├── action_engine/      # Dose optimization
│   ├── planner/            # Flight planning
│   └── api/                # FastAPI server
├── data/                   # Data directories
│   ├── raw/               # Raw datasets
│   ├── annotated/         # COCO format annotations
│   ├── masks/             # Segmentation masks
│   └── field/             # Field trial data
├── models/                 # Trained model artifacts
├── configs/                # Configuration files
├── tests/                  # Test suites
├── ui/                     # React frontend
├── deploy/                 # Deployment configurations
├── docs/                   # Documentation
└── logs/                   # System logs
```

## Development Environment

### Local Development

1. **Start the database:**
```bash
docker-compose up -d db redis
```

2. **Run the API server:**
```bash
python code/api/server.py
```

3. **Run the React UI:**
```bash
cd ui
npm start
```

4. **Access the application:**
- API: http://localhost:8000
- UI: http://localhost:3000
- API Docs: http://localhost:8000/docs

### Docker Development

```bash
# Build and start all services
docker-compose up --build

# Run specific services
docker-compose up api ui

# View logs
docker-compose logs -f api
```

## Training Models

### Data Preparation

1. **Download the dataset:**
```bash
python code/data_prep/download_kaggle.py
```

2. **Convert to COCO format:**
```bash
python code/data_prep/convert_to_coco.py
```

3. **Verify data:**
```bash
python -c "
import json
with open('data/annotated/instances_train.json') as f:
    data = json.load(f)
print(f'Training images: {len(data[\"images\"])}')
print(f'Training annotations: {len(data[\"annotations\"])}')
"
```

### Vision Model Training

1. **Train baseline detector:**
```bash
python code/vision/train_detector.py --config configs/yolov8_baseline.yaml
```

2. **Monitor training:**
```bash
# View training logs
tail -f logs/training/yolov8_baseline.log

# Access MLflow UI
mlflow ui --port 5000
```

3. **Evaluate model:**
```bash
python code/vision/evaluate_model.py --model models/yolov8_baseline/weights/best.pt
```

### Fusion Model Training

1. **Prepare fusion data:**
```bash
python code/fusion/prepare_fusion_data.py
```

2. **Train fusion model:**
```bash
python code/fusion/train_fusion.py --config configs/fusion_model.yaml
```

3. **Validate fusion model:**
```bash
python code/fusion/validate_fusion.py --model models/fusion_model.pt
```

## API Development

### Adding New Endpoints

1. **Create endpoint in `code/api/server.py`:**
```python
@app.post("/new-endpoint")
async def new_endpoint(request: NewRequestModel):
    """New endpoint description."""
    try:
        # Implementation
        return NewResponseModel(...)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

2. **Add request/response models:**
```python
class NewRequestModel(BaseModel):
    field1: str
    field2: Optional[int] = None

class NewResponseModel(BaseModel):
    result: str
    status: str
```

3. **Add tests:**
```python
def test_new_endpoint():
    response = client.post("/new-endpoint", json={"field1": "value"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
```

### API Testing

```bash
# Run API tests
pytest tests/unit/test_api.py -v

# Test specific endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg"
```

## Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_optimizer.py -v

# Run with coverage
pytest tests/unit/ --cov=code --cov-report=html
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v

# Test with Docker
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Performance Tests

```bash
# Run performance tests
locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 30s
```

### Test Data

```bash
# Generate test data
python tests/generate_test_data.py

# Validate test data
python tests/validate_test_data.py
```

## Deployment

### Local Deployment

1. **Build production images:**
```bash
docker-compose -f docker-compose.prod.yml build
```

2. **Start production services:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Verify deployment:**
```bash
curl http://localhost:8000/health
```

### Cloud Deployment

1. **Set up cloud infrastructure:**
```bash
# Using Terraform (example)
cd deploy/terraform
terraform init
terraform plan
terraform apply
```

2. **Deploy application:**
```bash
# Using Kubernetes
kubectl apply -f deploy/k8s/
```

3. **Monitor deployment:**
```bash
kubectl get pods
kubectl logs -f deployment/agrispray-api
```

### Edge Deployment

1. **Build edge image:**
```bash
docker build --target edge -t agrispray:edge .
```

2. **Deploy to Jetson:**
```bash
# Copy to Jetson device
scp agrispray:edge.tar jetson:/tmp/

# Load and run on Jetson
ssh jetson "docker load < /tmp/agrispray:edge.tar"
ssh jetson "docker run -d -p 8000:8000 agrispray:edge"
```

## Contributing

### Code Style

- **Python:** Follow PEP 8, use Black for formatting
- **JavaScript:** Follow ESLint rules, use Prettier for formatting
- **Documentation:** Use Markdown, follow Google style for docstrings

### Git Workflow

1. **Create feature branch:**
```bash
git checkout -b feature/new-feature
```

2. **Make changes and commit:**
```bash
git add .
git commit -m "feat: add new feature"
```

3. **Push and create PR:**
```bash
git push origin feature/new-feature
# Create pull request on GitHub
```

### Code Review Process

1. All code must pass CI/CD pipeline
2. At least one reviewer approval required
3. All tests must pass
4. Documentation must be updated

### Release Process

1. **Create release branch:**
```bash
git checkout -b release/v1.0.0
```

2. **Update version numbers:**
```bash
# Update version in package.json, setup.py, etc.
```

3. **Create release:**
```bash
git tag v1.0.0
git push origin v1.0.0
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce batch size in training config
   - Use gradient accumulation
   - Enable mixed precision training

2. **API connection errors:**
   - Check if services are running: `docker-compose ps`
   - Verify port availability: `netstat -tulpn | grep :8000`
   - Check logs: `docker-compose logs api`

3. **Model loading errors:**
   - Verify model file exists: `ls -la models/`
   - Check model compatibility: `python -c "import torch; print(torch.__version__)"`
   - Re-download models if corrupted

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python code/api/server.py

# Run with debugger
python -m pdb code/api/server.py
```

### Performance Profiling

```bash
# Profile Python code
python -m cProfile -o profile.stats code/api/server.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Profile memory usage
python -m memory_profiler code/api/server.py
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## Support

- **Issues:** Create GitHub issues for bugs and feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas
- **Documentation:** Check the docs/ directory for detailed guides
- **Community:** Join our Discord server for real-time help
