---
title: HomeCredit MLOps API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Home Credit MLOps Project

## Objective
Build an end-to-end MLOps pipeline to predict loan default risk for Home Credit. 
This project provides both an interactive web dashboard (Gradio) and a scalable REST API (FastAPI) for scoring inference.


## Structure
- `app.py`: Interactive Gradio dashboard deployed on Hugging Face Spaces.
- `api/`: FastAPI backend application for ML scoring, inference, logging, and profiling.
- `artifacts/`: Exported models (ONNX formats), preprocessing files (Joblib), and testing datasets (`inference_pool.json`).
- `src/`: Training pipelines and feature engineering scripts.
- `notebooks/`: Experiments, EDA, model training, and data drift analysis.
- `logs/`: Output files for system and prediction logs.
- `fluentd/`: Logging pipeline configuration to route logs to Elasticsearch.

## Stack
- **Machine Learning**: Scikit-Learn, LightGBM, XGBoost, ONNX Runtime
- **Tracking**: MLflow
- **Serving & UI**: FastAPI, Gradio
- **CI/CD**: GitHub Actions, Pytest, Hugging Face Spaces
- **Containerization & Monitoring**: Docker, Docker Compose, Elasticsearch, Grafana, Fluentd

## Getting Started Locally

### 1. Interactive Dashboard (Gradio)
```bash
python app.py
```
*Access the interface locally at `http://127.0.0.1:7860`.*

### 2. FastAPI Backend
```bash
uvicorn api.main:app --reload
```
*API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.*

### 3. Model Tracking (MLflow)
```bash
cd notebook/mlruns 
mlflow ui
```

## Docker Operations

### Standalone API
```bash
docker build --no-cache -t homecredit_scorer .
docker run -p 8000:8000 homecredit_scorer
```

### Full Monitoring Stack (API + ELK/EFK + Grafana)
```bash
docker-compose build --no-cache  # Build all service images
docker-compose up -d             # Start all containers in detached mode
docker ps                        # View running containers
docker-compose down              # Stop and remove containers
```


## CI/CD Pipeline
1. **Push**: Code is pushed to the GitHub repository.
2. **Test**: GitHub Actions automatically runs the tests using `pytest`.
3. **Build**: If the tests pass, a Docker image builds and a dry-run tests the container.
4. **Deploy**: The Gradio app is automatically deployed to Hugging Face Spaces ([MLops_2](https://huggingface.co/spaces/Fouxy84/MLops_2)).


