# Module 10: Model Deployment and Serving

## Overview
This module covers deploying ML models to production, including deployment patterns, serving frameworks, model registries, A/B testing, and monitoring.

## 1. Deployment Patterns

### Batch Inference
- Process data in batches
- Scheduled or on-demand
- Cost-effective for large volumes
- No real-time requirements

```python
def batch_inference(model, data_path, output_path):
    """Batch inference pipeline"""
    # Load data
    data = pd.read_parquet(data_path)
    
    # Preprocess
    processed = preprocess(data)
    
    # Predict
    predictions = model.predict(processed)
    
    # Post-process
    results = postprocess(predictions, data)
    
    # Save results
    results.to_parquet(output_path)
```

### Real-Time Inference
- Low latency requirements
- On-demand predictions
- REST/gRPC APIs
- Auto-scaling

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = torch.load('model.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_tensor = preprocess(data)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)
    
    return jsonify({'prediction': prediction.item()})
```

### Edge Deployment
- Deploy to edge devices
- Low latency, offline capability
- Model optimization required
- Limited resources

## 2. Model Serving Frameworks

### TorchServe
```python
# Create model archive
torch-model-archiver \
    --model-name plant_classifier \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pth \
    --handler handler.py \
    --extra-files index_to_name.json

# Start TorchServe
torchserve --start --model-store model_store --models plant_classifier.mar

# Make prediction
curl http://localhost:8080/predictions/plant_classifier \
    -T test_image.jpg
```

### TensorFlow Serving
```python
# Save model in SavedModel format
model.save('saved_model/plant_classifier/1/')

# Start TensorFlow Serving
tensorflow_model_server \
    --port=8500 \
    --rest_api_port=8501 \
    --model_name=plant_classifier \
    --model_base_path=/path/to/saved_model/plant_classifier

# Make prediction
curl -X POST http://localhost:8501/v1/models/plant_classifier:predict \
    -d '{"instances": [...]}'
```

### Triton Inference Server
```python
# Model repository structure
model_repository/
  plant_classifier/
    config.pbtxt
    1/
      model.plan  # TensorRT engine
```

## 3. Model Registry Systems

### MLflow Model Registry
```python
import mlflow
import mlflow.pytorch

# Register model
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="plant_classifier",
    registered_model_name="PlantClassifier"
)

# Transition to staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="PlantClassifier",
    version=1,
    stage="Staging"
)

# Load model from registry
model = mlflow.pytorch.load_model(
    model_uri="models:/PlantClassifier/Production"
)
```

## 4. Blue-Green and Canary Deployments

### Blue-Green Deployment
```python
def blue_green_deploy(new_model, traffic_percentage=100):
    """Blue-green deployment"""
    # Deploy new model (green)
    deploy_model(new_model, endpoint="green")
    
    # Route traffic gradually
    if traffic_percentage == 100:
        # Switch all traffic
        route_traffic(green=100, blue=0)
    else:
        # Gradual rollout
        route_traffic(green=traffic_percentage, blue=100-traffic_percentage)
    
    # Monitor green deployment
    if monitor_metrics(green)['error_rate'] < threshold:
        # Complete switch
        route_traffic(green=100, blue=0)
        # Remove old deployment
        remove_deployment("blue")
    else:
        # Rollback
        route_traffic(green=0, blue=100)
        remove_deployment("green")
```

### Canary Deployment
```python
def canary_deploy(new_model, initial_traffic=10):
    """Canary deployment with gradual rollout"""
    # Deploy canary
    deploy_model(new_model, endpoint="canary")
    
    # Start with small traffic
    route_traffic(canary=initial_traffic, production=100-initial_traffic)
    
    # Gradually increase
    for traffic in [10, 25, 50, 100]:
        route_traffic(canary=traffic, production=100-traffic)
        
        # Monitor
        metrics = monitor_metrics("canary")
        if metrics['error_rate'] > threshold:
            # Rollback
            route_traffic(canary=0, production=100)
            return False
        
        # Wait before next increase
        time.sleep(3600)  # 1 hour
    
    # Promote to production
    promote_to_production("canary")
    return True
```

## 5. A/B Testing for ML Models

### A/B Testing Framework
```python
import random
from datetime import datetime

class ABTest:
    """A/B testing for ML models"""
    
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.results = {'a': [], 'b': []}
    
    def predict(self, input_data, user_id=None):
        """Route to A or B based on user_id"""
        # Consistent assignment
        if user_id:
            assignment = hash(user_id) % 100 < (self.split_ratio * 100)
        else:
            assignment = random.random() < self.split_ratio
        
        if assignment:
            prediction = self.model_a.predict(input_data)
            variant = 'a'
        else:
            prediction = self.model_b.predict(input_data)
            variant = 'b'
        
        # Log result
        self.results[variant].append({
            'timestamp': datetime.now(),
            'input': input_data,
            'prediction': prediction
        })
        
        return prediction, variant
    
    def analyze_results(self):
        """Analyze A/B test results"""
        # Calculate metrics for each variant
        metrics_a = calculate_metrics(self.results['a'])
        metrics_b = calculate_metrics(self.results['b'])
        
        # Statistical significance test
        significance = statistical_test(metrics_a, metrics_b)
        
        return {
            'variant_a': metrics_a,
            'variant_b': metrics_b,
            'significance': significance,
            'winner': 'a' if metrics_a['accuracy'] > metrics_b['accuracy'] else 'b'
        }
```

## 6. Model Monitoring and Observability

### Performance Monitoring
```python
import time
from collections import deque

class ModelMonitor:
    """Monitor model performance"""
    
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
    
    def log_prediction(self, input_data, prediction, latency):
        """Log prediction"""
        self.predictions.append({
            'input': input_data,
            'prediction': prediction,
            'timestamp': time.time()
        })
        self.latencies.append(latency)
    
    def log_error(self, error):
        """Log error"""
        self.errors.append({
            'error': str(error),
            'timestamp': time.time()
        })
    
    def get_metrics(self):
        """Get current metrics"""
        return {
            'avg_latency': np.mean(self.latencies) if self.latencies else 0,
            'p95_latency': np.percentile(self.latencies, 95) if self.latencies else 0,
            'error_rate': len(self.errors) / len(self.predictions) if self.predictions else 0,
            'throughput': len(self.predictions) / (time.time() - self.predictions[0]['timestamp']) if self.predictions else 0
        }
```

### Data Drift Detection
```python
from scipy import stats

def detect_data_drift(reference_data, current_data, threshold=0.05):
    """Detect data drift using statistical tests"""
    drift_detected = {}
    
    for column in reference_data.columns:
        if reference_data[column].dtype in ['float64', 'int64']:
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                reference_data[column],
                current_data[column]
            )
            
            if p_value < threshold:
                drift_detected[column] = {
                    'p_value': p_value,
                    'statistic': statistic
                }
    
    return drift_detected
```

## 7. Production Troubleshooting

### Common Issues and Solutions

#### High Latency
```python
# Optimize model
model = quantize_model(model)
model = optimize_for_inference(model)

# Use caching
@lru_cache(maxsize=1000)
def cached_predict(input_hash):
    return model.predict(input)

# Batch processing
def batch_predict(model, inputs, batch_size=32):
    predictions = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        preds = model.predict(batch)
        predictions.extend(preds)
    return predictions
```

#### Memory Issues
```python
# Use model quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Clear cache
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

#### Model Degradation
```python
def monitor_model_performance(model, test_data, baseline_accuracy):
    """Monitor for model degradation"""
    current_accuracy = evaluate_model(model, test_data)
    
    if current_accuracy < baseline_accuracy * 0.95:  # 5% degradation
        alert("Model performance degraded!")
        trigger_retraining()
```

## 8. Best Practices

1. **Version Models**: Track model versions
2. **Monitor Performance**: Track latency, accuracy, errors
3. **Implement Rollback**: Quick rollback to previous version
4. **A/B Testing**: Test new models before full deployment
5. **Load Testing**: Test under expected load
6. **Error Handling**: Graceful error handling
7. **Logging**: Comprehensive logging
8. **Documentation**: Document deployment process
9. **Security**: Secure API endpoints
10. **Cost Optimization**: Monitor and optimize costs

## Next Steps

Continue to [Module 11: CI/CD for Machine Learning](11-ci-cd-ml.md) to learn about automating ML workflows.

