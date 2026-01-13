# Module 6: PySpark and DataBricks

## Overview
PySpark and DataBricks are essential for large-scale data processing in ML pipelines. This module covers Apache Spark fundamentals, PySpark operations, DataBricks platform, and integration with ML workflows.

## 1. Apache Spark Fundamentals

### What is Spark?
- Distributed computing framework
- In-memory processing for speed
- Fault-tolerant and scalable
- Supports batch and streaming

### Spark Architecture
- **Driver**: Coordinates job execution
- **Executors**: Run tasks on worker nodes
- **Cluster Manager**: Manages resources (YARN, Mesos, Kubernetes, Standalone)

### Spark Components
- **Spark Core**: Basic functionality
- **Spark SQL**: Structured data processing
- **Spark Streaming**: Real-time data processing
- **MLlib**: Machine learning library
- **GraphX**: Graph processing

## 2. PySpark Basics

### Setting Up PySpark
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Create SparkSession
spark = SparkSession.builder \
    .appName("PlantPhenotypingAnalysis") \
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
    .getOrCreate()

# Set log level
spark.sparkContext.setLogLevel("WARN")
```

### Creating DataFrames

#### From CSV
```python
# Read CSV
df = spark.read.csv(
    "s3://bucket/plant_data.csv",
    header=True,
    inferSchema=True
)

# With explicit schema
schema = StructType([
    StructField("plant_id", StringType(), True),
    StructField("leaf_area", FloatType(), True),
    StructField("height", FloatType(), True),
    StructField("disease_status", StringType(), True),
    StructField("timestamp", StringType(), True)
])

df = spark.read.csv(
    "s3://bucket/plant_data.csv",
    schema=schema,
    header=True
)
```

#### From Parquet
```python
# Read Parquet (preferred format)
df = spark.read.parquet("s3://bucket/plant_data.parquet")

# Read multiple files
df = spark.read.parquet("s3://bucket/plant_data/year=2024/month=*/")
```

#### From JSON
```python
df = spark.read.json("s3://bucket/plant_data.json")
```

#### From Database
```python
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://host:5432/dbname") \
    .option("dbtable", "plant_measurements") \
    .option("user", "username") \
    .option("password", "password") \
    .load()
```

### Basic DataFrame Operations
```python
# Select columns
df.select("plant_id", "leaf_area", "height").show()

# Filter
df.filter(col("disease_status") == "healthy").show()
df.filter((col("height") > 10) & (col("leaf_area") < 100)).show()

# Add column
df = df.withColumn("height_category", 
    when(col("height") < 10, "short")
    .when(col("height") < 20, "medium")
    .otherwise("tall")
)

# Rename column
df = df.withColumnRenamed("disease_status", "health_status")

# Drop column
df = df.drop("unnecessary_column")

# Distinct values
df.select("disease_status").distinct().show()

# Sort
df.orderBy(col("height").desc()).show()
```

### Aggregations
```python
# Group by and aggregate
df.groupBy("disease_status") \
    .agg(
        count("*").alias("count"),
        avg("height").alias("avg_height"),
        avg("leaf_area").alias("avg_leaf_area"),
        sum("leaf_area").alias("total_leaf_area")
    ) \
    .show()

# Window functions
from pyspark.sql.window import Window

window_spec = Window.partitionBy("plant_id").orderBy("timestamp")
df = df.withColumn("height_growth", 
    col("height") - lag("height", 1).over(window_spec)
)
```

### Joins
```python
# Inner join
df1.join(df2, on="plant_id", how="inner")

# Left join
df1.join(df2, on="plant_id", how="left")

# Multiple join keys
df1.join(df2, ["plant_id", "date"], how="inner")

# Join with different column names
df1.join(df2, df1.plant_id == df2.id, how="inner")
```

## 3. Advanced PySpark Operations

### User-Defined Functions (UDFs)
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Python function
def calculate_biomass_estimate(height, leaf_area):
    """Estimate biomass from height and leaf area"""
    return height * 0.5 + leaf_area * 0.3

# Register UDF
biomass_udf = udf(calculate_biomass_estimate, FloatType())

# Use UDF
df = df.withColumn("estimated_biomass", 
    biomass_udf(col("height"), col("leaf_area"))
)

# Vectorized UDF (faster)
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(returnType=FloatType())
def vectorized_biomass(height: pd.Series, leaf_area: pd.Series) -> pd.Series:
    return height * 0.5 + leaf_area * 0.3

df = df.withColumn("estimated_biomass", 
    vectorized_biomass(col("height"), col("leaf_area"))
)
```

### Handling Missing Data
```python
# Drop rows with nulls
df.na.drop()

# Drop rows with nulls in specific columns
df.na.drop(subset=["height", "leaf_area"])

# Fill nulls
df.na.fill(0)  # Fill with 0
df.na.fill({"height": 0, "leaf_area": 0})  # Fill specific columns
df.na.fill({"height": df.select(avg("height")).first()[0]})  # Fill with mean

# Forward fill
from pyspark.sql.window import Window
window = Window.partitionBy("plant_id").orderBy("timestamp")
df = df.withColumn("height_filled", 
    last("height", ignorenulls=True).over(window)
)
```

### Partitioning and Bucketing
```python
# Repartition
df = df.repartition(10)  # 10 partitions
df = df.repartition("disease_status")  # Partition by column

# Coalesce (reduce partitions)
df = df.coalesce(5)

# Write partitioned data
df.write \
    .partitionBy("year", "month") \
    .parquet("s3://bucket/plant_data_partitioned/")

# Bucketing
df.write \
    .bucketBy(10, "plant_id") \
    .saveAsTable("plant_data_bucketed")
```

### Caching
```python
# Cache DataFrame
df.cache()

# Persist with storage level
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# Unpersist
df.unpersist()

# Check if cached
df.is_cached
```

## 4. Spark SQL

### SQL Queries
```python
# Register DataFrame as table
df.createOrReplaceTempView("plants")

# Run SQL query
result = spark.sql("""
    SELECT 
        disease_status,
        AVG(height) as avg_height,
        AVG(leaf_area) as avg_leaf_area,
        COUNT(*) as count
    FROM plants
    WHERE height > 10
    GROUP BY disease_status
    ORDER BY avg_height DESC
""")

result.show()
```

### Complex SQL Operations
```python
# Window functions in SQL
spark.sql("""
    SELECT 
        plant_id,
        timestamp,
        height,
        AVG(height) OVER (
            PARTITION BY plant_id 
            ORDER BY timestamp 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as moving_avg_height
    FROM plants
""").show()

# CTEs (Common Table Expressions)
spark.sql("""
    WITH healthy_plants AS (
        SELECT * FROM plants WHERE disease_status = 'healthy'
    ),
    tall_plants AS (
        SELECT * FROM healthy_plants WHERE height > 15
    )
    SELECT 
        AVG(leaf_area) as avg_leaf_area
    FROM tall_plants
""").show()
```

## 5. DataBricks Platform

### DataBricks Overview
- Unified analytics platform
- Built on Apache Spark
- Collaborative notebooks
- Integrated MLflow
- Delta Lake support

### Setting Up DataBricks
```python
# DataBricks automatically provides SparkSession
# Access it as 'spark'

# Read from DataBricks File System (DBFS)
df = spark.read.parquet("/dbfs/mnt/plant_data/")

# Or from cloud storage mounted to DBFS
df = spark.read.parquet("/mnt/plant-data/raw/")
```

### DataBricks Notebooks
```python
# Magic commands
%python  # Python code
%sql     # SQL code
%scala   # Scala code
%r       # R code

# File system operations
%fs ls /mnt/plant-data/

# Display data
display(df)  # Interactive visualization
```

### Delta Lake
```python
# Read Delta table
df = spark.read.format("delta").load("/mnt/plant-data/delta/")

# Write Delta table
df.write.format("delta").mode("overwrite").save("/mnt/plant-data/delta/")

# Create Delta table
df.write.format("delta").saveAsTable("plant_measurements")

# Time travel (query previous versions)
df = spark.read.format("delta").option("versionAsOf", 0).load("/mnt/plant-data/delta/")

# Optimize Delta table
spark.sql("OPTIMIZE delta.`/mnt/plant-data/delta/`")

# Z-order clustering
spark.sql("OPTIMIZE delta.`/mnt/plant-data/delta/` ZORDER BY (plant_id, timestamp)")

# Vacuum old files
spark.sql("VACUUM delta.`/mnt/plant-data/delta/` RETAIN 168 HOURS")
```

## 6. DataBricks Workflows

### Creating Workflows
```python
# Workflow definition (in DataBricks UI or via API)
# Task 1: Data ingestion
# Task 2: Data transformation
# Task 3: Feature engineering
# Task 4: Model training
# Task 5: Model inference
```

### Workflow Dependencies
```python
# Using DataBricks Jobs API
import requests

def create_job(job_config):
    """Create DataBricks job"""
    response = requests.post(
        "https://your-workspace.cloud.databricks.com/api/2.0/jobs/create",
        headers={"Authorization": f"Bearer {token}"},
        json=job_config
    )
    return response.json()

# Job configuration
job_config = {
    "name": "Plant Phenotyping Pipeline",
    "tasks": [
        {
            "task_key": "ingest_data",
            "notebook_task": {
                "notebook_path": "/Workspace/ingest_data"
            }
        },
        {
            "task_key": "transform_data",
            "notebook_task": {
                "notebook_path": "/Workspace/transform_data"
            },
            "depends_on": [{"task_key": "ingest_data"}]
        },
        {
            "task_key": "train_model",
            "notebook_task": {
                "notebook_path": "/Workspace/train_model"
            },
            "depends_on": [{"task_key": "transform_data"}]
        }
    ]
}
```

## 7. Performance Optimization

### Optimization Strategies
```python
# 1. Use appropriate file formats (Parquet, Delta)
df.write.parquet("output.parquet")  # Columnar, compressed

# 2. Partition data
df.write.partitionBy("year", "month").parquet("output/")

# 3. Use broadcast joins for small tables
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), "key")

# 4. Cache frequently used DataFrames
df.cache()

# 5. Optimize joins
# - Put smaller DataFrame on right side
# - Use broadcast for small tables
# - Ensure join keys are partitioned

# 6. Tune Spark configuration
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

### Monitoring Performance
```python
# Spark UI: http://driver-node:4040
# Check:
# - Job execution time
# - Stage details
# - Task distribution
# - Shuffle read/write
# - Memory usage

# Get execution plan
df.explain(True)

# Check partitions
df.rdd.getNumPartitions()
```

## 8. Integration with ML Pipelines

### Feature Engineering with PySpark
```python
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

# Assemble features
assembler = VectorAssembler(
    inputCols=["height", "leaf_area", "width"],
    outputCol="features"
)
df_features = assembler.transform(df)

# Scale features
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)
scaler_model = scaler.fit(df_features)
df_scaled = scaler_model.transform(df_features)

# PCA
pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)
```

### MLlib Models
```python
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

# Classification
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="disease_status",
    numTrees=100
)

# Regression
rf_reg = RandomForestRegressor(
    featuresCol="features",
    labelCol="height",
    numTrees=100
)

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
```

## 9. Complete Example: Plant Phenotyping Pipeline

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, count, sum
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Initialize Spark
spark = SparkSession.builder \
    .appName("PlantPhenotypingPipeline") \
    .getOrCreate()

# 1. Read data
df = spark.read.parquet("/mnt/plant-data/raw/")

# 2. Data cleaning
df_clean = df.filter(col("height") > 0) \
    .filter(col("leaf_area") > 0) \
    .na.drop(subset=["height", "leaf_area"])

# 3. Feature engineering
df_features = df_clean.withColumn(
    "height_category",
    when(col("height") < 10, "short")
    .when(col("height") < 20, "medium")
    .otherwise("tall")
)

# 4. Prepare for ML
assembler = VectorAssembler(
    inputCols=["height", "leaf_area", "width"],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)

classifier = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="disease_status",
    numTrees=100
)

# 5. Create pipeline
pipeline = Pipeline(stages=[assembler, scaler, classifier])

# 6. Train
train_df, test_df = df_features.randomSplit([0.8, 0.2])
model = pipeline.fit(train_df)

# 7. Evaluate
predictions = model.transform(test_df)
predictions.select("disease_status", "prediction", "probability").show()

# 8. Save model
model.write().overwrite().save("/mnt/models/plant_disease_classifier")

# 9. Save results
predictions.write.mode("overwrite").parquet("/mnt/plant-data/predictions/")
```

## 10. Best Practices

1. **Use Parquet/Delta**: Columnar formats are faster
2. **Partition Data**: Partition by date or category
3. **Avoid Collect**: Use `show()` or write to storage instead
4. **Cache Wisely**: Only cache DataFrames used multiple times
5. **Optimize Joins**: Use broadcast for small tables
6. **Monitor Resources**: Watch memory and CPU usage
7. **Use Delta Lake**: For ACID transactions and time travel
8. **Test Locally**: Test on small datasets first

## Next Steps

Proceed to [Module 7: Database Management](07-database-management.md) to learn SQL, PL/SQL, and Snowflake for ML applications.

