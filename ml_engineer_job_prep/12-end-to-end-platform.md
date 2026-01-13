# Module 12: End-to-End ML Platform

## Overview
This module integrates all concepts into a complete ML platform architecture, covering data management, feature stores, model lifecycle, monitoring, and team collaboration workflows.

## 1. ML Platform Architecture

### Platform Components
```
ML Platform
├── Data Layer
│   ├── Data Ingestion
│   ├── Data Storage (Data Lake/Warehouse)
│   ├── Data Versioning (DVC)
│   └── Data Quality
├── Feature Layer
│   ├── Feature Store
│   ├── Feature Engineering
│   └── Feature Serving
├── Model Layer
│   ├── Training Infrastructure
│   ├── Experiment Tracking
│   ├── Model Registry
│   └── Model Versioning
├── Serving Layer
│   ├── Model Serving
│   ├── API Gateway
│   ├── Load Balancing
│   └── Auto-scaling
└── Operations Layer
    ├── Monitoring
    ├── Logging
    ├── Alerting
    └── CI/CD
```

## 2. DataOps Integration

### Data Pipeline Orchestration
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

def data_ops_pipeline():
    """Complete data operations pipeline"""
    dag = DAG('data_ops_pipeline', schedule_interval='@daily')
    
    # Data ingestion
    ingest_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_from_sources,
        dag=dag
    )
    
    # Data validation
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_quality,
        dag=dag
    )
    
    # Data transformation
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        dag=dag
    )
    
    # Feature engineering
    features_task = PythonOperator(
        task_id='create_features',
        python_callable=create_features,
        dag=dag
    )
    
    # Store features
    store_task = PythonOperator(
        task_id='store_features',
        python_callable=store_to_feature_store,
        dag=dag
    )
    
    ingest_task >> validate_task >> transform_task >> features_task >> store_task
```

## 3. Feature Store

### Feature Store Implementation
```python
from feast import FeatureStore, Entity, Feature, ValueType
from feast.data_source import FileSource

# Define entities
plant_entity = Entity(
    name="plant",
    value_type=ValueType.STRING,
    description="Plant identifier"
)

# Define features
plant_features = [
    Feature(name="height", dtype=ValueType.FLOAT),
    Feature(name="leaf_area", dtype=ValueType.FLOAT),
    Feature(name="disease_status", dtype=ValueType.STRING),
]

# Define data source
plant_data_source = FileSource(
    path="s3://bucket/features/plant_features.parquet",
    timestamp_field="event_timestamp"
)

# Create feature view
from feast import FeatureView

plant_feature_view = FeatureView(
    name="plant_features",
    entities=[plant_entity],
    features=plant_features,
    source=plant_data_source,
    ttl=timedelta(days=365)
)

# Initialize feature store
fs = FeatureStore(repo_path=".")

# Register features
fs.apply([plant_entity, plant_feature_view])

# Retrieve features
features = fs.get_online_features(
    entity_rows=[{"plant": "PLANT001"}],
    features=["plant_features:height", "plant_features:leaf_area"]
)
```

## 4. Model Lifecycle Management

### Complete Lifecycle
```python
class ModelLifecycle:
    """Manage complete model lifecycle"""
    
    def __init__(self):
        self.stages = {
            'development': self.development_stage,
            'staging': self.staging_stage,
            'production': self.production_stage,
            'deprecated': self.deprecated_stage
        }
    
    def development_stage(self, model):
        """Development stage"""
        # Train and validate
        model = train_model()
        metrics = evaluate_model(model)
        
        # Register if meets criteria
        if metrics['accuracy'] > 0.85:
            register_model(model, stage='development')
        
        return model
    
    def staging_stage(self, model):
        """Staging stage"""
        # Deploy to staging
        deploy_model(model, environment='staging')
        
        # Run integration tests
        run_integration_tests(model)
        
        # A/B test
        ab_test_results = run_ab_test(model)
        
        if ab_test_results['success']:
            promote_to_production(model)
    
    def production_stage(self, model):
        """Production stage"""
        # Deploy to production
        deploy_model(model, environment='production')
        
        # Monitor
        monitor_model(model)
        
        # Check for retraining
        if should_retrain(model):
            trigger_retraining()
    
    def deprecated_stage(self, model):
        """Deprecate model"""
        # Stop serving
        stop_serving(model)
        
        # Archive
        archive_model(model)
```

## 5. Monitoring and Observability

### Comprehensive Monitoring
```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
data_drift_score = Gauge('data_drift_score', 'Data drift score')

class MonitoringSystem:
    """Comprehensive monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
    
    def log_prediction(self, input_data, prediction, latency):
        """Log prediction"""
        prediction_counter.inc()
        prediction_latency.observe(latency)
        
        # Store for analysis
        self.metrics_collector.store_prediction(
            input_data, prediction, latency
        )
    
    def check_data_drift(self):
        """Check for data drift"""
        score = detect_data_drift()
        data_drift_score.set(score)
        
        if score > threshold:
            self.alert_manager.send_alert("Data drift detected!")
    
    def check_model_performance(self):
        """Check model performance"""
        accuracy = calculate_current_accuracy()
        model_accuracy.set(accuracy)
        
        if accuracy < baseline_accuracy * 0.95:
            self.alert_manager.send_alert("Model performance degraded!")
            trigger_retraining()
```

## 6. Cost Management

### Cost Optimization
```python
class CostManager:
    """Manage and optimize ML platform costs"""
    
    def __init__(self):
        self.cost_tracker = CostTracker()
    
    def track_training_costs(self, job_id, resources):
        """Track training job costs"""
        cost = calculate_training_cost(resources)
        self.cost_tracker.record_cost('training', job_id, cost)
    
    def track_inference_costs(self, endpoint, requests):
        """Track inference costs"""
        cost = calculate_inference_cost(endpoint, requests)
        self.cost_tracker.record_cost('inference', endpoint, cost)
    
    def optimize_costs(self):
        """Optimize platform costs"""
        # Analyze costs
        cost_analysis = self.cost_tracker.analyze()
        
        # Recommendations
        recommendations = []
        
        # Right-size resources
        if cost_analysis['training']['overprovisioned']:
            recommendations.append("Reduce training instance sizes")
        
        # Use spot instances
        if cost_analysis['training']['spot_eligible']:
            recommendations.append("Use spot instances for training")
        
        # Batch inference
        if cost_analysis['inference']['real_time_heavy']:
            recommendations.append("Move to batch inference where possible")
        
        return recommendations
```

## 7. Security and Compliance

### Security Measures
```python
class SecurityManager:
    """Manage platform security"""
    
    def __init__(self):
        self.access_control = AccessControl()
        self.data_encryption = DataEncryption()
        self.audit_logger = AuditLogger()
    
    def authenticate_user(self, user_id, credentials):
        """Authenticate user"""
        if self.access_control.verify(user_id, credentials):
            self.audit_logger.log_access(user_id, 'authenticated')
            return True
        return False
    
    def authorize_access(self, user_id, resource):
        """Authorize resource access"""
        if self.access_control.has_permission(user_id, resource):
            self.audit_logger.log_access(user_id, f'accessed {resource}')
            return True
        return False
    
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive data"""
        return self.data_encryption.encrypt(data)
    
    def audit_log(self, event):
        """Log security events"""
        self.audit_logger.log_security_event(event)
```

## 8. Team Collaboration Workflows

### Collaboration Tools
```python
class CollaborationWorkflow:
    """Team collaboration workflows"""
    
    def __init__(self):
        self.notification_service = NotificationService()
        self.documentation = DocumentationSystem()
    
    def notify_model_ready(self, model_id, stage):
        """Notify team when model is ready"""
        message = f"Model {model_id} is ready for {stage}"
        self.notification_service.send_to_team(message)
    
    def request_review(self, model_id, reviewer):
        """Request model review"""
        self.notification_service.send_to_user(
            reviewer,
            f"Please review model {model_id}"
        )
    
    def document_experiment(self, experiment_id, results):
        """Document experiment"""
        self.documentation.create_experiment_doc(
            experiment_id, results
        )
    
    def share_insights(self, insights):
        """Share insights with team"""
        self.documentation.add_insights(insights)
        self.notification_service.send_to_team(
            "New insights available"
        )
```

## 9. Complete Platform Example

### Platform Integration
```python
class MLPlatform:
    """Complete ML platform"""
    
    def __init__(self, config):
        self.config = config
        self.data_ops = DataOpsPipeline()
        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        self.serving_layer = ServingLayer()
        self.monitoring = MonitoringSystem()
        self.collaboration = CollaborationWorkflow()
    
    def run_pipeline(self):
        """Run complete ML pipeline"""
        # 1. Data operations
        data = self.data_ops.run()
        
        # 2. Feature engineering
        features = self.feature_store.create_features(data)
        
        # 3. Model training
        model = train_model(features)
        
        # 4. Register model
        model_id = self.model_registry.register(model)
        
        # 5. Deploy model
        endpoint = self.serving_layer.deploy(model_id)
        
        # 6. Monitor
        self.monitoring.start_monitoring(endpoint)
        
        # 7. Notify team
        self.collaboration.notify_model_ready(model_id, 'production')
        
        return endpoint
```

## 10. Best Practices

1. **Modularity**: Design platform with modular components
2. **Scalability**: Design for horizontal scaling
3. **Reliability**: Implement fault tolerance
4. **Security**: Secure all components
5. **Monitoring**: Comprehensive monitoring and alerting
6. **Documentation**: Document all components
7. **Testing**: Test all components thoroughly
8. **Cost Management**: Monitor and optimize costs
9. **Team Collaboration**: Enable efficient collaboration
10. **Continuous Improvement**: Iterate and improve platform

## Summary

This tutorial has covered:
- Plant phenotyping fundamentals and deep learning
- Data engineering and ETL pipelines
- Database management (SQL, PL/SQL, Snowflake)
- PySpark and DataBricks
- Cloud deployment (Azure, GCP)
- CI/CD for ML
- Model development and deployment
- End-to-end ML platform

You now have the knowledge to be a top candidate for ML Engineer roles!

