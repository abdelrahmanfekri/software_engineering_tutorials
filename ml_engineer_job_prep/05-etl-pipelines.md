# Module 5: ETL Pipelines and Data Processing

## Overview
This module covers building scalable, production-ready ETL (Extract, Transform, Load) pipelines for ML applications, including orchestration, error handling, monitoring, and best practices.

## 1. ETL vs ELT Patterns

### ETL (Extract, Transform, Load)
- Transform data before loading into destination
- Good for: Structured transformations, data quality enforcement
- Tools: Apache Airflow, Prefect, Luigi

### ELT (Extract, Load, Transform)
- Load raw data first, transform in destination
- Good for: Large datasets, cloud data warehouses
- Tools: dbt, SQL-based transformations

### Choosing the Right Pattern
```python
# ETL: Transform in Python before loading
def etl_pipeline():
    # Extract
    raw_data = extract_from_source()
    
    # Transform
    transformed_data = transform_data(raw_data)
    
    # Load
    load_to_destination(transformed_data)

# ELT: Load raw, transform in SQL
def elt_pipeline():
    # Extract and Load
    load_raw_data_to_warehouse()
    
    # Transform (in SQL/dbt)
    run_sql_transformations()
```

## 2. Building Scalable Data Pipelines

### Pipeline Architecture
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

class PipelineStage(ABC):
    """Base class for pipeline stages"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stage logic"""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate stage output"""
        return True

class ExtractStage(PipelineStage):
    """Extract data from source"""
    
    def __init__(self, source_config: Dict):
        super().__init__("Extract")
        self.source_config = source_config
    
    def execute(self, context: Dict) -> Dict:
        self.logger.info(f"Extracting from {self.source_config['type']}")
        
        if self.source_config['type'] == 'api':
            data = self._extract_from_api()
        elif self.source_config['type'] == 'database':
            data = self._extract_from_db()
        elif self.source_config['type'] == 'file':
            data = self._extract_from_file()
        else:
            raise ValueError(f"Unknown source type: {self.source_config['type']}")
        
        context['raw_data'] = data
        return context
    
    def _extract_from_api(self):
        import requests
        response = requests.get(self.source_config['url'])
        return response.json()
    
    def _extract_from_db(self):
        import pandas as pd
        from sqlalchemy import create_engine
        engine = create_engine(self.source_config['connection_string'])
        return pd.read_sql(self.source_config['query'], engine)
    
    def _extract_from_file(self):
        import pandas as pd
        return pd.read_parquet(self.source_config['path'])

class TransformStage(PipelineStage):
    """Transform data"""
    
    def __init__(self, transformations: List[callable]):
        super().__init__("Transform")
        self.transformations = transformations
    
    def execute(self, context: Dict) -> Dict:
        data = context['raw_data']
        
        for transform in self.transformations:
            self.logger.info(f"Applying transformation: {transform.__name__}")
            data = transform(data)
        
        context['transformed_data'] = data
        return context

class LoadStage(PipelineStage):
    """Load data to destination"""
    
    def __init__(self, destination_config: Dict):
        super().__init__("Load")
        self.destination_config = destination_config
    
    def execute(self, context: Dict) -> Dict:
        data = context['transformed_data']
        
        if self.destination_config['type'] == 'database':
            self._load_to_db(data)
        elif self.destination_config['type'] == 'file':
            self._load_to_file(data)
        elif self.destination_config['type'] == 'data_lake':
            self._load_to_data_lake(data)
        
        context['loaded'] = True
        return context
    
    def _load_to_db(self, data):
        from sqlalchemy import create_engine
        engine = create_engine(self.destination_config['connection_string'])
        data.to_sql(
            self.destination_config['table'],
            engine,
            if_exists='append',
            index=False
        )
    
    def _load_to_file(self, data):
        data.to_parquet(self.destination_config['path'], index=False)
    
    def _load_to_data_lake(self, data):
        import boto3
        s3 = boto3.client('s3')
        data.to_parquet(f"s3://{self.destination_config['bucket']}/{self.destination_config['key']}")

class DataPipeline:
    """Complete ETL pipeline"""
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self.logger = logging.getLogger("Pipeline")
    
    def run(self, initial_context: Dict = None) -> Dict:
        """Run pipeline"""
        context = initial_context or {}
        
        try:
            for stage in self.stages:
                self.logger.info(f"Running stage: {stage.name}")
                context = stage.execute(context)
                
                # Validate
                if not stage.validate(context.get('transformed_data')):
                    raise ValueError(f"Validation failed for stage: {stage.name}")
            
            self.logger.info("Pipeline completed successfully")
            return context
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
```

## 3. Pipeline Orchestration with Airflow

### Airflow DAG
```python
# dags/plant_phenotyping_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'plant_phenotyping_pipeline',
    default_args=default_args,
    description='ETL pipeline for plant phenotyping data',
    schedule_interval=timedelta(hours=6),
    catchup=False
)

def extract_data():
    """Extract data from sources"""
    from scripts.extract import extract_plant_data
    extract_plant_data()

def transform_data():
    """Transform extracted data"""
    from scripts.transform import transform_plant_data
    transform_plant_data()

def load_data():
    """Load transformed data"""
    from scripts.load import load_plant_data
    load_plant_data()

def validate_data():
    """Validate loaded data"""
    from scripts.validate import validate_plant_data
    validate_plant_data()

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

# Define dependencies
extract_task >> transform_task >> load_task >> validate_task
```

### Airflow with Sensors
```python
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.sql import SqlSensor

# Wait for file
file_sensor = FileSensor(
    task_id='wait_for_data_file',
    filepath='/data/raw/plant_data.csv',
    poke_interval=60,
    timeout=3600,
    dag=dag
)

# Wait for database condition
db_sensor = SqlSensor(
    task_id='wait_for_new_data',
    conn_id='postgres_default',
    sql="SELECT COUNT(*) FROM plant_measurements WHERE created_at > CURRENT_DATE - INTERVAL '1 day'",
    poke_interval=300,
    timeout=3600,
    dag=dag
)

file_sensor >> extract_task
```

## 4. Prefect for Dynamic Workflows

### Prefect Flow
```python
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def extract_data(source: str):
    """Extract data with caching"""
    # Extraction logic
    return data

@task
def transform_data(data):
    """Transform data"""
    # Transformation logic
    return transformed_data

@task
def load_data(data, destination: str):
    """Load data"""
    # Loading logic
    pass

@flow(name="plant_phenotyping_pipeline")
def pipeline_flow(source: str, destination: str):
    """Main pipeline flow"""
    # Extract
    raw_data = extract_data(source)
    
    # Transform
    transformed_data = transform_data(raw_data)
    
    # Load
    load_data(transformed_data, destination)
    
    return "Pipeline completed"

# Run flow
if __name__ == "__main__":
    pipeline_flow(source="s3://bucket/raw/", destination="s3://bucket/processed/")
```

## 5. Error Handling and Retry Logic

### Retry Decorator
```python
import time
from functools import wraps
from typing import Callable

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    
                    wait_time = delay * (backoff ** (attempt - 1))
                    print(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        return wrapper
    return decorator

@retry(max_attempts=5, delay=2.0, backoff=2.0)
def extract_with_retry(source: str):
    """Extract with automatic retry"""
    # Extraction logic that might fail
    return extract_data(source)
```

### Circuit Breaker Pattern
```python
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half_open'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half_open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            
            raise
```

## 6. Data Validation in Pipelines

### Schema Validation
```python
from pydantic import BaseModel, ValidationError
from typing import List, Optional

class PlantMeasurement(BaseModel):
    """Schema for plant measurement"""
    plant_id: str
    measurement_date: str
    height: float
    leaf_area: float
    disease_status: str
    
    class Config:
        extra = 'forbid'  # Reject extra fields

def validate_schema(data: List[dict]) -> List[dict]:
    """Validate data against schema"""
    validated_data = []
    errors = []
    
    for i, record in enumerate(data):
        try:
            validated = PlantMeasurement(**record)
            validated_data.append(validated.dict())
        except ValidationError as e:
            errors.append({
                'index': i,
                'record': record,
                'errors': e.errors()
            })
    
    if errors:
        logging.warning(f"Validation errors: {errors}")
    
    return validated_data
```

### Data Quality Checks
```python
def check_data_quality(df):
    """Perform data quality checks"""
    checks = {
        'completeness': check_completeness(df),
        'uniqueness': check_uniqueness(df),
        'validity': check_validity(df),
        'consistency': check_consistency(df)
    }
    
    # Fail if any check fails
    if not all(checks.values()):
        raise ValueError(f"Data quality checks failed: {checks}")
    
    return checks

def check_completeness(df, threshold=0.95):
    """Check data completeness"""
    completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
    return completeness >= threshold

def check_uniqueness(df, key_columns):
    """Check uniqueness of key columns"""
    return df[key_columns].duplicated().sum() == 0

def check_validity(df, constraints):
    """Check value validity"""
    for col, constraint in constraints.items():
        if 'min' in constraint and df[col].min() < constraint['min']:
            return False
        if 'max' in constraint and df[col].max() > constraint['max']:
            return False
    return True

def check_consistency(df):
    """Check data consistency"""
    # Example: height should be positive
    if 'height' in df.columns:
        return (df['height'] > 0).all()
    return True
```

## 7. Monitoring and Alerting

### Pipeline Monitoring
```python
import time
from datetime import datetime
from typing import Dict, Any

class PipelineMonitor:
    """Monitor pipeline execution"""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'records_processed': 0,
            'errors': []
        }
    
    def start(self):
        """Start monitoring"""
        self.metrics['start_time'] = datetime.now()
    
    def end(self):
        """End monitoring"""
        self.metrics['end_time'] = datetime.now()
        self.metrics['duration'] = (
            self.metrics['end_time'] - self.metrics['start_time']
        ).total_seconds()
    
    def record_processed(self, count: int):
        """Record processed records"""
        self.metrics['records_processed'] += count
    
    def record_error(self, error: Exception):
        """Record error"""
        self.metrics['errors'].append({
            'timestamp': datetime.now(),
            'error': str(error)
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return self.metrics
    
    def send_alert(self, threshold: float = 3600):
        """Send alert if pipeline takes too long"""
        if self.metrics['duration'] and self.metrics['duration'] > threshold:
            # Send alert (email, Slack, etc.)
            print(f"ALERT: Pipeline took {self.metrics['duration']}s")
```

## 8. Best Practices

1. **Idempotency**: Pipelines should be rerunnable safely
2. **Error Handling**: Implement comprehensive error handling
3. **Logging**: Log all operations for debugging
4. **Monitoring**: Monitor pipeline health and performance
5. **Testing**: Test pipelines with sample data
6. **Documentation**: Document all transformations
7. **Versioning**: Version pipeline code and configurations
8. **Scalability**: Design for horizontal scaling
9. **Security**: Secure credentials and sensitive data
10. **Cost Optimization**: Optimize for cloud costs

## Next Steps

Continue to [Module 6: PySpark and DataBricks](06-pyspark-databricks.md) for large-scale data processing.

