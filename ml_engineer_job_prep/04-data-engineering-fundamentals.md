# Module 4: Data Engineering Fundamentals

## Overview
Data engineering is the foundation of successful ML systems. This module covers data ingestion, wrangling, cleaning, validation, and management - essential skills for pulling, stitching, and preparing data from various sources for ML applications.

## 1. Understanding Data Sources

### First-Party Data Sources
- Internal databases (PostgreSQL, MySQL, MongoDB)
- Application logs and events
- IoT sensors and devices
- Internal APIs and services
- File systems and data lakes

### Third-Party Data Sources
- Public APIs (REST, GraphQL)
- External databases
- Cloud storage (S3, GCS, Azure Blob)
- Data vendors and marketplaces
- Social media platforms
- Government datasets

### Data Source Challenges
- **Format variety**: JSON, CSV, Parquet, Avro, XML
- **Schema evolution**: Changing data structures
- **Rate limits**: API throttling
- **Authentication**: OAuth, API keys, certificates
- **Data quality**: Missing values, inconsistencies
- **Volume**: Large-scale data handling

## 2. Data Pulling and Ingestion Strategies

### API Data Pulling
```python
import requests
import time
from typing import List, Dict
import pandas as pd

class APIDataPuller:
    """Pull data from REST APIs"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def pull_data(self, endpoint: str, params: Dict = None, 
                  max_retries: int = 3) -> List[Dict]:
        """Pull data with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/{endpoint}",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def pull_paginated_data(self, endpoint: str, page_size: int = 100):
        """Pull paginated data"""
        all_data = []
        page = 1
        
        while True:
            data = self.pull_data(endpoint, params={'page': page, 'size': page_size})
            if not data:
                break
            all_data.extend(data)
            if len(data) < page_size:
                break
            page += 1
            time.sleep(0.1)  # Rate limiting
        
        return all_data
```

### Database Data Pulling
```python
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

class DatabaseDataPuller:
    """Pull data from databases"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
    
    def pull_table(self, table_name: str, query: str = None, 
                   chunksize: int = 10000) -> pd.DataFrame:
        """Pull entire table or query result"""
        if query:
            sql = query
        else:
            sql = f"SELECT * FROM {table_name}"
        
        # Use chunks for large tables
        chunks = []
        for chunk in pd.read_sql(sql, self.engine, chunksize=chunksize):
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def pull_incremental(self, table_name: str, timestamp_column: str, 
                         last_timestamp: str) -> pd.DataFrame:
        """Pull incremental data based on timestamp"""
        query = f"""
        SELECT * FROM {table_name}
        WHERE {timestamp_column} > '{last_timestamp}'
        ORDER BY {timestamp_column}
        """
        return pd.read_sql(query, self.engine)
```

### File-Based Data Pulling
```python
import boto3
import pandas as pd
from pathlib import Path

class FileDataPuller:
    """Pull data from files (local, S3, GCS, Azure)"""
    
    def __init__(self, source_type: str = 'local', **kwargs):
        self.source_type = source_type
        if source_type == 's3':
            self.s3_client = boto3.client('s3', **kwargs)
        elif source_type == 'gcs':
            from google.cloud import storage
            self.gcs_client = storage.Client(**kwargs)
    
    def pull_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Pull CSV file"""
        if self.source_type == 'local':
            return pd.read_csv(file_path, **kwargs)
        elif self.source_type == 's3':
            bucket, key = file_path.replace('s3://', '').split('/', 1)
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return pd.read_csv(obj['Body'], **kwargs)
    
    def pull_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Pull Parquet file"""
        if self.source_type == 'local':
            return pd.read_parquet(file_path, **kwargs)
        elif self.source_type == 's3':
            bucket, key = file_path.replace('s3://', '').split('/', 1)
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(obj['Body'], **kwargs)
    
    def pull_directory(self, directory_path: str, pattern: str = '*.csv'):
        """Pull all files matching pattern from directory"""
        files = []
        if self.source_type == 'local':
            files = list(Path(directory_path).glob(pattern))
        elif self.source_type == 's3':
            bucket, prefix = directory_path.replace('s3://', '').split('/', 1)
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            files = [f"s3://{bucket}/{obj['Key']}" 
                    for obj in response.get('Contents', []) 
                    if obj['Key'].endswith(pattern.replace('*', ''))]
        
        dataframes = []
        for file in files:
            if file.endswith('.csv'):
                dataframes.append(self.pull_csv(file))
            elif file.endswith('.parquet'):
                dataframes.append(self.pull_parquet(file))
        
        return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
```

## 3. Data Wrangling and Cleaning

### Data Cleaning Functions
```python
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class DataCleaner:
    """Comprehensive data cleaning utilities"""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop',
                             columns: List[str] = None) -> pd.DataFrame:
        """Handle missing values"""
        if columns is None:
            columns = df.columns
        
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=columns)
        elif strategy == 'fill_mean':
            for col in columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'fill_median':
            for col in columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'fill_mode':
            for col in columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        elif strategy == 'forward_fill':
            df_clean[columns].fillna(method='ffill', inplace=True)
        elif strategy == 'backward_fill':
            df_clean[columns].fillna(method='bfill', inplace=True)
        
        return df_clean
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: List[str] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates(subset=subset, keep=keep)
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, columns: List[str],
                       method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers using IQR or Z-score"""
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / 
                                 df_clean[col].std())
                df_clean = df_clean[z_scores < 3]
        
        return df_clean
    
    @staticmethod
    def normalize_text(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize text columns"""
        df_clean = df.copy()
        for col in columns:
            df_clean[col] = df_clean[col].str.lower().str.strip()
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        return df_clean
    
    @staticmethod
    def convert_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
        """Convert column types"""
        df_clean = df.copy()
        for col, dtype in type_mapping.items():
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].astype(dtype)
                except ValueError as e:
                    print(f"Error converting {col} to {dtype}: {e}")
        return df_clean
```

### Data Transformation
```python
class DataTransformer:
    """Data transformation utilities"""
    
    @staticmethod
    def parse_json_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Parse JSON column into separate columns"""
        import json
        df_clean = df.copy()
        
        # Parse JSON
        parsed = df_clean[column].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        
        # Expand into columns
        expanded = pd.json_normalize(parsed)
        expanded.columns = [f"{column}_{c}" for c in expanded.columns]
        
        # Merge back
        df_clean = pd.concat([df_clean.drop(columns=[column]), expanded], axis=1)
        return df_clean
    
    @staticmethod
    def extract_datetime_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Extract features from datetime column"""
        df_clean = df.copy()
        df_clean[column] = pd.to_datetime(df_clean[column])
        
        df_clean[f'{column}_year'] = df_clean[column].dt.year
        df_clean[f'{column}_month'] = df_clean[column].dt.month
        df_clean[f'{column}_day'] = df_clean[column].dt.day
        df_clean[f'{column}_dayofweek'] = df_clean[column].dt.dayofweek
        df_clean[f'{column}_hour'] = df_clean[column].dt.hour
        
        return df_clean
    
    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str],
                          method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical variables"""
        df_clean = df.copy()
        
        if method == 'onehot':
            df_clean = pd.get_dummies(df_clean, columns=columns, prefix=columns)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                df_clean[col] = le.fit_transform(df_clean[col])
        
        return df_clean
```

## 4. Data Filtering, Tagging, and Normalization

### Data Filtering
```python
class DataFilter:
    """Data filtering utilities"""
    
    @staticmethod
    def filter_by_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Filter using pandas query"""
        return df.query(condition)
    
    @staticmethod
    def filter_by_date_range(df: pd.DataFrame, date_column: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """Filter by date range"""
        df[date_column] = pd.to_datetime(df[date_column])
        mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
        return df[mask]
    
    @staticmethod
    def filter_by_value_range(df: pd.DataFrame, column: str,
                             min_val: float, max_val: float) -> pd.DataFrame:
        """Filter by value range"""
        return df[(df[column] >= min_val) & (df[column] <= max_val)]
    
    @staticmethod
    def filter_by_list(df: pd.DataFrame, column: str,
                      values: List[Any]) -> pd.DataFrame:
        """Filter rows where column value is in list"""
        return df[df[column].isin(values)]
```

### Data Tagging
```python
class DataTagger:
    """Add tags/metadata to data"""
    
    @staticmethod
    def add_quality_tags(df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality tags"""
        df_tagged = df.copy()
        
        # Tag missing values
        df_tagged['_has_missing'] = df_tagged.isnull().any(axis=1)
        
        # Tag outliers
        numeric_cols = df_tagged.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df_tagged[col].quantile(0.25)
            Q3 = df_tagged[col].quantile(0.75)
            IQR = Q3 - Q1
            df_tagged[f'_{col}_is_outlier'] = (
                (df_tagged[col] < Q1 - 1.5 * IQR) | 
                (df_tagged[col] > Q3 + 1.5 * IQR)
            )
        
        # Tag duplicates
        df_tagged['_is_duplicate'] = df_tagged.duplicated()
        
        return df_tagged
    
    @staticmethod
    def add_metadata_tags(df: pd.DataFrame, source: str,
                         ingestion_date: str = None) -> pd.DataFrame:
        """Add metadata tags"""
        df_tagged = df.copy()
        df_tagged['_data_source'] = source
        df_tagged['_ingestion_date'] = pd.Timestamp.now() if ingestion_date is None else ingestion_date
        df_tagged['_row_id'] = range(len(df_tagged))
        return df_tagged
```

### Data Normalization
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataNormalizer:
    """Data normalization utilities"""
    
    @staticmethod
    def standardize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Standardize to mean=0, std=1"""
        df_norm = df.copy()
        scaler = StandardScaler()
        df_norm[columns] = scaler.fit_transform(df_norm[columns])
        return df_norm, scaler
    
    @staticmethod
    def minmax_scale(df: pd.DataFrame, columns: List[str],
                    range: tuple = (0, 1)) -> pd.DataFrame:
        """Scale to range [min, max]"""
        df_norm = df.copy()
        scaler = MinMaxScaler(feature_range=range)
        df_norm[columns] = scaler.fit_transform(df_norm[columns])
        return df_norm, scaler
    
    @staticmethod
    def robust_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Robust scaling using median and IQR"""
        df_norm = df.copy()
        scaler = RobustScaler()
        df_norm[columns] = scaler.fit_transform(df_norm[columns])
        return df_norm, scaler
```

## 5. Data Joining and Stitching

### Data Joining
```python
class DataJoiner:
    """Join data from multiple sources"""
    
    @staticmethod
    def join_dataframes(df1: pd.DataFrame, df2: pd.DataFrame,
                       join_keys: Dict[str, str], how: str = 'inner') -> pd.DataFrame:
        """Join two dataframes"""
        left_key, right_key = join_keys['left'], join_keys['right']
        return df1.merge(df2, left_on=left_key, right_on=right_key, how=how)
    
    @staticmethod
    def join_multiple(dfs: List[pd.DataFrame], join_keys: List[str],
                     how: str = 'inner') -> pd.DataFrame:
        """Join multiple dataframes"""
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on=join_keys, how=how)
        return result
    
    @staticmethod
    def stitch_time_series(dfs: List[pd.DataFrame], 
                          timestamp_column: str) -> pd.DataFrame:
        """Stitch time series data"""
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(timestamp_column)
        combined = combined.drop_duplicates(subset=[timestamp_column], keep='last')
        return combined
```

## 6. Data Quality Validation

### Schema Validation
```python
from pydantic import BaseModel, ValidationError
from typing import Optional

class DataValidator:
    """Data quality validation"""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame, schema: Dict[str, type]) -> bool:
        """Validate dataframe against schema"""
        for col, expected_type in schema.items():
            if col not in df.columns:
                print(f"Missing column: {col}")
                return False
            if not df[col].dtype == expected_type:
                print(f"Type mismatch for {col}: expected {expected_type}, got {df[col].dtype}")
                return False
        return True
    
    @staticmethod
    def check_completeness(df: pd.DataFrame, threshold: float = 0.95) -> bool:
        """Check if data completeness meets threshold"""
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        return completeness >= threshold
    
    @staticmethod
    def check_uniqueness(df: pd.DataFrame, key_columns: List[str]) -> bool:
        """Check uniqueness of key columns"""
        return df[key_columns].duplicated().sum() == 0
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame, 
                          constraints: Dict[str, Dict[str, float]]) -> bool:
        """Check value ranges"""
        for col, constraints_dict in constraints.items():
            if 'min' in constraints_dict:
                if df[col].min() < constraints_dict['min']:
                    return False
            if 'max' in constraints_dict:
                if df[col].max() > constraints_dict['max']:
                    return False
        return True
```

## 7. Data Management Principles

### Data Cataloging
```python
class DataCatalog:
    """Data catalog for tracking datasets"""
    
    def __init__(self):
        self.catalog = {}
    
    def register_dataset(self, name: str, metadata: Dict):
        """Register dataset in catalog"""
        self.catalog[name] = {
            'name': name,
            'registered_at': pd.Timestamp.now(),
            **metadata
        }
    
    def get_dataset_info(self, name: str) -> Dict:
        """Get dataset information"""
        return self.catalog.get(name, {})
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets"""
        return list(self.catalog.keys())
```

### Data Lineage Tracking
```python
class DataLineage:
    """Track data lineage"""
    
    def __init__(self):
        self.lineage = {}
    
    def track_transformation(self, output_name: str, input_names: List[str],
                           transformation: str):
        """Track data transformation"""
        self.lineage[output_name] = {
            'inputs': input_names,
            'transformation': transformation,
            'timestamp': pd.Timestamp.now()
        }
    
    def get_lineage(self, dataset_name: str) -> Dict:
        """Get lineage for dataset"""
        return self.lineage.get(dataset_name, {})
```

## 8. Complete Data Pipeline Example

```python
class DataPipeline:
    """Complete data pipeline"""
    
    def __init__(self):
        self.puller = APIDataPuller('https://api.example.com')
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.validator = DataValidator()
        self.catalog = DataCatalog()
    
    def run(self, source: str, destination: str):
        """Run complete pipeline"""
        # 1. Pull data
        print("Pulling data...")
        raw_data = self.puller.pull_data(source)
        df = pd.DataFrame(raw_data)
        
        # 2. Clean data
        print("Cleaning data...")
        df_clean = self.cleaner.handle_missing_values(df, strategy='fill_mean')
        df_clean = self.cleaner.remove_duplicates(df_clean)
        
        # 3. Transform data
        print("Transforming data...")
        df_transformed = self.transformer.extract_datetime_features(
            df_clean, 'timestamp'
        )
        
        # 4. Validate data
        print("Validating data...")
        if not self.validator.check_completeness(df_transformed):
            raise ValueError("Data completeness check failed")
        
        # 5. Save data
        print("Saving data...")
        df_transformed.to_parquet(destination, index=False)
        
        # 6. Register in catalog
        self.catalog.register_dataset(
            destination,
            {
                'source': source,
                'rows': len(df_transformed),
                'columns': list(df_transformed.columns),
                'schema': df_transformed.dtypes.to_dict()
            }
        )
        
        return df_transformed
```

## 9. Best Practices

1. **Idempotency**: Ensure pipelines can be rerun safely
2. **Error Handling**: Implement robust error handling and retries
3. **Logging**: Log all operations for debugging
4. **Monitoring**: Monitor data quality and pipeline health
5. **Documentation**: Document data schemas and transformations
6. **Testing**: Test data transformations with sample data
7. **Versioning**: Version your data and transformations

## Next Steps

Proceed to [Module 5: ETL Pipelines and Data Processing](05-etl-pipelines.md) to learn about building scalable, production-ready data pipelines.

