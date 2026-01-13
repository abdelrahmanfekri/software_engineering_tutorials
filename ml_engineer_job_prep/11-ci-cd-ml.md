# Module 11: CI/CD for Machine Learning

## Overview
Continuous Integration and Continuous Deployment (CI/CD) for ML automates testing, validation, and deployment of ML models. This module covers GitHub Actions, GitLab CI/CD, Jenkins, and best practices for ML pipelines.

## 1. CI/CD Fundamentals for ML

### Why CI/CD for ML?
- **Reproducibility**: Consistent model training and deployment
- **Quality**: Automated testing catches issues early
- **Speed**: Faster iteration cycles
- **Reliability**: Automated validation gates
- **Collaboration**: Team members can deploy safely

### ML-Specific CI/CD Challenges
- Large model files
- Data dependencies
- Long training times
- Model validation requirements
- A/B testing needs
- Rollback strategies

## 2. GitHub Actions for ML

### Basic Workflow
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### ML-Specific Workflow
```yaml
# .github/workflows/ml-training.yml
name: ML Training Pipeline

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'config/**'
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'Experiment name'
        required: true

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Validate data
        run: |
          python scripts/validate_data.py \
            --data-path s3://bucket/data/raw/ \
            --schema-path schemas/data_schema.json
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

  train-model:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Train model
        run: |
          python src/train.py \
            --data-path s3://bucket/data/processed/ \
            --output-path s3://bucket/models/ \
            --experiment-name ${{ github.event.inputs.experiment_name || 'default' }}
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Evaluate model
        run: |
          python src/evaluate.py \
            --model-path s3://bucket/models/latest/ \
            --test-data s3://bucket/data/test/
      
      - name: Model validation gate
        run: |
          python scripts/validate_model.py \
            --model-path s3://bucket/models/latest/ \
            --min-accuracy 0.85 \
            --max-latency 100

  deploy-staging:
    needs: train-model
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to staging
        run: |
          python scripts/deploy.py \
            --model-path s3://bucket/models/latest/ \
            --environment staging \
            --endpoint-name plant-classifier-staging
        env:
          AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
          AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}
          AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}

  integration-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run integration tests
        run: |
          pytest tests/integration/ \
            --endpoint-url ${{ secrets.STAGING_ENDPOINT_URL }} \
            --api-key ${{ secrets.STAGING_API_KEY }}

  deploy-production:
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        run: |
          python scripts/deploy.py \
            --model-path s3://bucket/models/latest/ \
            --environment production \
            --endpoint-name plant-classifier-prod \
            --traffic-percentage 10
        env:
          AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
          AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}
          AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
```

### Matrix Strategy for Testing
```yaml
# .github/workflows/test-matrix.yml
name: Test Matrix

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
        os: [ubuntu-latest, windows-latest]
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/
```

## 3. GitLab CI/CD for ML

### GitLab CI Configuration
```yaml
# .gitlab-ci.yml
stages:
  - validate
  - test
  - train
  - deploy

variables:
  PYTHON_VERSION: "3.9"
  DOCKER_IMAGE: "python:${PYTHON_VERSION}"

validate-data:
  stage: validate
  image: ${DOCKER_IMAGE}
  script:
    - pip install -r requirements.txt
    - python scripts/validate_data.py
  only:
    - main
    - develop

unit-tests:
  stage: test
  image: ${DOCKER_IMAGE}
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest tests/unit/ --cov=src --cov-report=term
  coverage: '/TOTAL.*\s+(\d+%)$/'

integration-tests:
  stage: test
  image: ${DOCKER_IMAGE}
  script:
    - pip install -r requirements.txt
    - pytest tests/integration/
  only:
    - main

train-model:
  stage: train
  image: ${DOCKER_IMAGE}
  script:
    - pip install -r requirements.txt
    - python src/train.py
  artifacts:
    paths:
      - models/
    expire_in: 1 week
  only:
    - main

deploy-staging:
  stage: deploy
  image: ${DOCKER_IMAGE}
  script:
    - pip install -r requirements.txt
    - python scripts/deploy.py --environment staging
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - main

deploy-production:
  stage: deploy
  image: ${DOCKER_IMAGE}
  script:
    - pip install -r requirements.txt
    - python scripts/deploy.py --environment production
  environment:
    name: production
    url: https://api.example.com
  when: manual
  only:
    - main
```

## 4. Jenkins for ML Pipelines

### Jenkinsfile
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        AWS_REGION = 'us-east-1'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup') {
            steps {
                sh '''
                    python${PYTHON_VERSION} -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Lint') {
            steps {
                sh '''
                    . venv/bin/activate
                    flake8 src/ --max-line-length=100
                    black --check src/
                '''
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest tests/ --cov=src --cov-report=xml
                '''
            }
            post {
                always {
                    publishCoverage adapters: [coberturaAdapter('coverage.xml')]
                }
            }
        }
        
        stage('Validate Data') {
            steps {
                sh '''
                    . venv/bin/activate
                    python scripts/validate_data.py
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                sh '''
                    . venv/bin/activate
                    python src/train.py \
                        --data-path s3://bucket/data/ \
                        --output-path s3://bucket/models/
                '''
            }
        }
        
        stage('Evaluate Model') {
            steps {
                sh '''
                    . venv/bin/activate
                    python src/evaluate.py \
                        --model-path s3://bucket/models/latest/
                '''
            }
        }
        
        stage('Model Validation') {
            steps {
                script {
                    def accuracy = sh(
                        script: '''
                            . venv/bin/activate
                            python scripts/get_model_metrics.py \
                                --model-path s3://bucket/models/latest/
                        ''',
                        returnStdout: true
                    ).trim()
                    
                    def minAccuracy = 0.85
                    if (accuracy.toFloat() < minAccuracy) {
                        error("Model accuracy ${accuracy} is below minimum ${minAccuracy}")
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh '''
                    . venv/bin/activate
                    python scripts/deploy.py --environment staging
                '''
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest tests/integration/ \
                        --endpoint-url ${STAGING_ENDPOINT_URL}
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                sh '''
                    . venv/bin/activate
                    python scripts/deploy.py --environment production
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            emailext (
                subject: "Pipeline Success: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build successful!",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
        failure {
            emailext (
                subject: "Pipeline Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## 5. Automated Testing for ML

### Unit Tests
```python
# tests/unit/test_preprocessing.py
import pytest
import numpy as np
from src.preprocessing import preprocess_image, normalize_features

def test_preprocess_image():
    """Test image preprocessing"""
    # Create dummy image
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Assertions
    assert processed.shape == (1, 224, 224, 3)
    assert processed.min() >= 0
    assert processed.max() <= 1

def test_normalize_features():
    """Test feature normalization"""
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    normalized = normalize_features(features)
    
    assert np.allclose(normalized.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(normalized.std(axis=0), 1, atol=1e-6)
```

### Model Tests
```python
# tests/unit/test_model.py
import pytest
import torch
from src.model import PlantClassifier

def test_model_forward():
    """Test model forward pass"""
    model = PlantClassifier(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    
    output = model(x)
    
    assert output.shape == (1, 10)
    assert torch.allclose(torch.sum(torch.softmax(output, dim=1)), torch.tensor(1.0))

def test_model_prediction():
    """Test model prediction"""
    model = PlantClassifier(num_classes=10)
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1)
    
    assert prediction.item() in range(10)
```

### Integration Tests
```python
# tests/integration/test_api.py
import pytest
import requests
import json

@pytest.fixture
def api_url():
    return "https://staging-api.example.com"

def test_health_endpoint(api_url):
    """Test health endpoint"""
    response = requests.get(f"{api_url}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(api_url):
    """Test prediction endpoint"""
    # Load test image
    with open("tests/data/test_image.jpg", "rb") as f:
        files = {"image": f}
        response = requests.post(f"{api_url}/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data
```

## 6. Model Validation Gates

### Validation Script
```python
# scripts/validate_model.py
import argparse
import json
import sys
from pathlib import Path

def validate_model(model_path, min_accuracy, max_latency):
    """Validate model meets requirements"""
    # Load model metrics
    metrics_path = Path(model_path) / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Check accuracy
    accuracy = metrics.get("accuracy", 0)
    if accuracy < min_accuracy:
        print(f"ERROR: Accuracy {accuracy} is below minimum {min_accuracy}")
        return False
    
    # Check latency
    latency = metrics.get("avg_latency_ms", float('inf'))
    if latency > max_latency:
        print(f"ERROR: Latency {latency}ms exceeds maximum {max_latency}ms")
        return False
    
    # Check model size
    model_size_mb = metrics.get("model_size_mb", 0)
    max_size_mb = 500
    if model_size_mb > max_size_mb:
        print(f"WARNING: Model size {model_size_mb}MB exceeds {max_size_mb}MB")
    
    print("Model validation passed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--max-latency", type=float, default=100)
    
    args = parser.parse_args()
    
    if not validate_model(args.model_path, args.min_accuracy, args.max_latency):
        sys.exit(1)
```

## 7. Infrastructure as Code

### Terraform for ML Infrastructure
```hcl
# infrastructure/main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Azure ML Workspace
resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = "plant-phenotyping-ws"
  location            = "eastus"
  resource_group_name = azurerm_resource_group.rg.name
  application_insights_id = azurerm_application_insights.insights.id
  key_vault_id        = azurerm_key_vault.kv.id
  storage_account_id  = azurerm_storage_account.storage.id
}

# Compute cluster
resource "azurerm_machine_learning_compute_cluster" "compute" {
  name                          = "gpu-cluster"
  location                      = "eastus"
  vm_priority                    = "Dedicated"
  vm_size                       = "Standard_NC6s_v3"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml_workspace.id
  
  scale_settings {
    min_node_count = 0
    max_node_count = 4
  }
}

# Online endpoint
resource "azurerm_machine_learning_online_endpoint" "endpoint" {
  name                = "plant-classifier-endpoint"
  location            = "eastus"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml_workspace.id
  auth_mode          = "key"
}
```

## 8. Continuous Training

### Automated Retraining Pipeline
```yaml
# .github/workflows/retrain.yml
name: Continuous Training

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check for new data
        run: |
          python scripts/check_new_data.py \
            --data-path s3://bucket/data/raw/ \
            --last-check-path s3://bucket/metadata/last_check.txt
      
      - name: Train new model
        if: steps.check.outputs.has_new_data == 'true'
        run: |
          python src/train.py \
            --data-path s3://bucket/data/processed/ \
            --output-path s3://bucket/models/
      
      - name: Compare models
        run: |
          python scripts/compare_models.py \
            --current-model s3://bucket/models/latest/ \
            --new-model s3://bucket/models/new/
      
      - name: Deploy if better
        if: steps.compare.outputs.is_better == 'true'
        run: |
          python scripts/deploy.py \
            --model-path s3://bucket/models/new/ \
            --environment production
```

## 9. Best Practices

1. **Test Early and Often**: Run tests on every commit
2. **Validate Data**: Check data quality before training
3. **Model Validation Gates**: Enforce minimum performance thresholds
4. **Staging Environment**: Always test in staging before production
5. **Rollback Strategy**: Plan for model rollbacks
6. **Monitor Deployments**: Track model performance post-deployment
7. **Documentation**: Document all pipeline steps
8. **Security**: Secure secrets and credentials
9. **Cost Management**: Monitor and optimize CI/CD costs
10. **Parallel Execution**: Run independent jobs in parallel

## Next Steps

Review [Module 12: End-to-End ML Platform](12-end-to-end-platform.md) for a complete platform architecture combining all concepts.

