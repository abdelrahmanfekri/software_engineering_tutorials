# Module 8: Cloud Deployment - Azure and GCP

## Overview
This module covers deploying ML models on Microsoft Azure and Google Cloud Platform, including containerization, orchestration, auto-scaling, and cost optimization strategies.

## 1. Azure ML Services

### Azure ML Overview
- Managed ML platform
- Automated ML capabilities
- Model registry and versioning
- Endpoint deployment
- MLOps integration

### Setting Up Azure ML
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Authenticate
credential = DefaultAzureCredential()

# Create ML client
ml_client = MLClient(
    credential=credential,
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace"
)
```

### Training Models on Azure
```python
from azure.ai.ml import command
from azure.ai.ml import Input, Output

# Define training job
job = command(
    code="./src",
    command="python train.py --data ${{inputs.data}} --model ${{outputs.model}}",
    inputs={
        "data": Input(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/plant_data")
    },
    outputs={
        "model": Output(type="uri_folder")
    },
    environment="azureml:plant-phenotyping-env:1",
    compute="gpu-cluster",
    experiment_name="plant-phenotyping",
    display_name="Train Plant Classifier"
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
```

### Deploying Models on Azure
```python
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# Register model
model = Model(
    path="./models/plant_classifier.pkl",
    name="plant-classifier",
    description="Plant disease classifier",
    type=AssetTypes.CUSTOM_MODEL
)
ml_client.models.create_or_update(model)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="plant-classifier-endpoint",
    description="Endpoint for plant disease classification",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint)

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="plant-classifier-endpoint",
    model="plant-classifier:1",
    environment="azureml:plant-phenotyping-env:1",
    code_path="./src",
    scoring_script="score.py",
    instance_type="Standard_DS2_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment)

# Allocate traffic
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint)
```

### Scoring Script
```python
# score.py
import json
import numpy as np
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('plant-classifier')
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)
        data = np.array(data['data'])
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

### Azure Container Instances
```python
from azure.ai.ml.entities import Model, Environment, CodeConfiguration
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# Deploy to ACI (for testing)
deployment = ManagedOnlineDeployment(
    name="aci-deployment",
    endpoint_name="plant-classifier-endpoint",
    model="plant-classifier:1",
    environment="azureml:plant-phenotyping-env:1",
    code_path="./src",
    scoring_script="score.py",
    instance_type="Standard_DS2_v2",
    instance_count=1
)
```

### Azure Kubernetes Service (AKS)
```python
# Attach AKS compute
from azure.ai.ml.entities import KubernetesCompute

aks_compute = KubernetesCompute(
    name="aks-cluster",
    namespace="default",
    resource_id="/subscriptions/.../resourceGroups/.../providers/Microsoft.ContainerService/managedClusters/aks-cluster"
)
ml_client.compute.begin_create_or_update(aks_compute)

# Deploy to AKS
deployment = ManagedOnlineDeployment(
    name="aks-deployment",
    endpoint_name="plant-classifier-endpoint",
    model="plant-classifier:1",
    environment="azureml:plant-phenotyping-env:1",
    code_path="./src",
    scoring_script="score.py",
    instance_count=3  # Multiple instances for scaling
)
```

## 2. Google Cloud Platform (GCP)

### Vertex AI Overview
- Unified ML platform
- AutoML capabilities
- Custom training
- Model deployment
- MLOps pipelines

### Setting Up Vertex AI
```python
from google.cloud import aiplatform
from google.oauth2 import service_account

# Initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file(
    'path/to/service-account-key.json'
)

aiplatform.init(
    project="your-project-id",
    location="us-central1",
    credentials=credentials
)
```

### Training Models on Vertex AI
```python
from google.cloud import aiplatform
from google.cloud.aiplatform import training_jobs

# Custom training job
job = aiplatform.CustomTrainingJob(
    display_name="plant-phenotyping-training",
    script_path="train.py",
    container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.1-9:latest",
    model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-9:latest",
    requirements=["torch", "torchvision", "pandas", "numpy"],
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)

# Run training
model = job.run(
    dataset=aiplatform.datasets.ImageDataset("your-dataset-id"),
    model_display_name="plant-classifier",
    args=["--epochs", "50", "--batch-size", "32"]
)
```

### Deploying Models on Vertex AI
```python
from google.cloud.aiplatform import models, endpoints

# Deploy to endpoint
endpoint = endpoints.Endpoint.create(
    display_name="plant-classifier-endpoint"
)

# Deploy model
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="plant-classifier-v1",
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3,
    traffic_percentage=100
)

# Make prediction
predictions = endpoint.predict(
    instances=[
        {
            "image_bytes": base64_encoded_image,
            "mime_type": "image/jpeg"
        }
    ]
)
```

### Vertex AI Pipelines
```python
from kfp.v2 import dsl
from kfp.v2.dsl import component, pipeline, Input, Output, Dataset, Model

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn"]
)
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(input_data.path)
    # Preprocessing logic
    df.to_csv(output_data.path, index=False)

@component(
    base_image="python:3.9",
    packages_to_install=["torch", "torchvision"]
)
def train_model(
    training_data: Input[Dataset],
    model: Output[Model]
):
    import torch
    # Training logic
    # Save model to model.path

@pipeline(name="plant-phenotyping-pipeline")
def plant_phenotyping_pipeline():
    preprocess_op = preprocess_data(input_data=raw_data)
    train_op = train_model(training_data=preprocess_op.outputs["output_data"])

# Compile and run
from kfp.v2 import compiler
compiler.Compiler().compile(
    pipeline_func=plant_phenotyping_pipeline,
    package_path="pipeline.json"
)

job = aiplatform.PipelineJob(
    display_name="plant-phenotyping-pipeline",
    template_path="pipeline.json",
    pipeline_root="gs://your-bucket/pipelines"
)
job.run()
```

## 3. Containerization with Docker

### Dockerfile for ML Models
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "src/app.py"]
```

### Docker Compose for Local Testing
```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/plant_classifier.pkl
    volumes:
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Flask API for Model Serving
```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = joblib.load('models/plant_classifier.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Preprocess image
        image_array = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(image_array)
        probabilities = model.predict_proba(image_array)
        
        return jsonify({
            "prediction": prediction.tolist(),
            "probabilities": probabilities.tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def preprocess_image(image):
    # Resize, normalize, etc.
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 224, 224, 3)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## 4. Kubernetes Orchestration

### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: plant-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: plant-classifier
  template:
    metadata:
      labels:
        app: plant-classifier
    spec:
      containers:
      - name: classifier
        image: gcr.io/your-project/plant-classifier:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        env:
        - name: MODEL_PATH
          value: "/app/models/plant_classifier.pkl"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: plant-classifier-service
spec:
  selector:
    app: plant-classifier
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Horizontal Pod Autoscaling
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: plant-classifier-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: plant-classifier
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 5. Auto-Scaling and Load Balancing

### Azure Auto-Scaling
```python
# Configure auto-scaling in Azure ML
deployment = ManagedOnlineDeployment(
    name="auto-scaled-deployment",
    endpoint_name="plant-classifier-endpoint",
    model="plant-classifier:1",
    environment="azureml:plant-phenotyping-env:1",
    code_path="./src",
    scoring_script="score.py",
    instance_type="Standard_DS2_v2",
    instance_count=1,
    # Auto-scaling configuration
    scale_settings={
        "min_instances": 1,
        "max_instances": 10,
        "target_utilization": 70
    }
)
```

### GCP Auto-Scaling
```python
# Configure auto-scaling in Vertex AI
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="plant-classifier-v1",
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100,
    # Auto-scaling based on requests
    autoscaling_target_cpu_utilization=70
)
```

## 6. Cost Optimization Strategies

### Right-Sizing Resources
```python
# Choose appropriate instance types
# For CPU-bound: Standard instances
# For GPU-bound: GPU instances (only when needed)

# Use spot instances for training
job = command(
    # ... other parameters
    resources={
        "instance_type": "Standard_NC6s_v3",
        "instance_count": 1,
        "use_spot": True  # Use spot instances for cost savings
    }
)
```

### Batch Inference
```python
# Use batch endpoints for cost-effective inference
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment

# Create batch endpoint
batch_endpoint = BatchEndpoint(
    name="plant-classifier-batch",
    description="Batch inference endpoint"
)
ml_client.batch_endpoints.begin_create_or_update(batch_endpoint)

# Create batch deployment
batch_deployment = BatchDeployment(
    name="batch-deployment",
    endpoint_name="plant-classifier-batch",
    model="plant-classifier:1",
    compute="cpu-cluster",
    instance_count=2
)
ml_client.batch_deployments.begin_create_or_update(batch_deployment)

# Submit batch job
job = ml_client.batch_endpoints.invoke(
    endpoint_name="plant-classifier-batch",
    deployment_name="batch-deployment",
    inputs={"data": Input(type="uri_folder", path="s3://bucket/batch-data")}
)
```

### Model Optimization
```python
# Quantize model for smaller size and faster inference
import torch
import torch.quantization

# Quantize PyTorch model
model_fp32 = torch.load('plant_classifier.pth')
model_fp32.eval()

# Prepare for quantization
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_int8 = torch.quantization.prepare(model_fp32)
# Calibrate with sample data
model_int8 = torch.quantization.convert(model_int8)

# Save quantized model
torch.save(model_int8, 'plant_classifier_quantized.pth')
```

## 7. Monitoring and Observability

### Azure ML Monitoring
```python
from azure.ai.ml.entities import OnlineEndpoint

# Enable Application Insights
endpoint = ManagedOnlineEndpoint(
    name="plant-classifier-endpoint",
    # ... other parameters
    app_insights_enabled=True
)
```

### GCP Monitoring
```python
from google.cloud import monitoring_v3

# Create monitoring client
client = monitoring_v3.MetricServiceClient()
project_name = f"projects/your-project-id"

# Create custom metric
descriptor = monitoring_v3.MetricDescriptor()
descriptor.type = "custom.googleapis.com/ml/prediction_latency"
descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DOUBLE
descriptor.description = "Prediction latency in milliseconds"

client.create_metric_descriptor(
    name=project_name, metric_descriptor=descriptor
)
```

## 8. Best Practices

1. **Use managed services** when possible (Azure ML, Vertex AI)
2. **Containerize applications** for portability
3. **Implement health checks** for reliability
4. **Set up auto-scaling** for cost efficiency
5. **Monitor performance** and costs
6. **Use batch inference** for non-real-time needs
7. **Optimize models** (quantization, pruning)
8. **Implement retry logic** for resilience
9. **Use staging environments** before production
10. **Document deployment processes**

## Next Steps

Continue to [Module 11: CI/CD for Machine Learning](11-ci-cd-ml.md) to learn about automating ML workflows with continuous integration and deployment.

