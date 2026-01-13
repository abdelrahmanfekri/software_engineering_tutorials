# Module 7: Database Management for ML Engineers

## Overview
Database management is crucial for ML engineers working with structured data. This module covers SQL fundamentals, advanced queries, PL/SQL programming, Snowflake data warehouse, and database integration with ML pipelines.

## 1. SQL Fundamentals

### Basic Queries
```sql
-- SELECT statement
SELECT plant_id, height, leaf_area, disease_status
FROM plant_measurements
WHERE height > 10
ORDER BY height DESC;

-- Aggregations
SELECT 
    disease_status,
    COUNT(*) as count,
    AVG(height) as avg_height,
    AVG(leaf_area) as avg_leaf_area,
    MIN(height) as min_height,
    MAX(height) as max_height,
    SUM(leaf_area) as total_leaf_area
FROM plant_measurements
GROUP BY disease_status
HAVING COUNT(*) > 100;

-- Joins
SELECT 
    p.plant_id,
    p.height,
    p.leaf_area,
    m.measurement_date,
    m.measurement_type
FROM plants p
INNER JOIN measurements m ON p.plant_id = m.plant_id
WHERE p.disease_status = 'healthy';

-- Subqueries
SELECT plant_id, height
FROM plant_measurements
WHERE height > (
    SELECT AVG(height) FROM plant_measurements
);
```

### Advanced SQL

#### Window Functions
```sql
-- Ranking
SELECT 
    plant_id,
    height,
    ROW_NUMBER() OVER (ORDER BY height DESC) as rank,
    RANK() OVER (ORDER BY height DESC) as rank_with_ties,
    DENSE_RANK() OVER (ORDER BY height DESC) as dense_rank
FROM plant_measurements;

-- Partitioned window functions
SELECT 
    plant_id,
    measurement_date,
    height,
    AVG(height) OVER (
        PARTITION BY plant_id 
        ORDER BY measurement_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_height,
    LAG(height, 1) OVER (PARTITION BY plant_id ORDER BY measurement_date) as prev_height
FROM plant_measurements;

-- Cumulative sums
SELECT 
    plant_id,
    measurement_date,
    leaf_area,
    SUM(leaf_area) OVER (
        PARTITION BY plant_id 
        ORDER BY measurement_date
    ) as cumulative_leaf_area
FROM plant_measurements;
```

#### Common Table Expressions (CTEs)
```sql
WITH healthy_plants AS (
    SELECT * FROM plant_measurements
    WHERE disease_status = 'healthy'
),
tall_plants AS (
    SELECT * FROM healthy_plants
    WHERE height > 15
),
avg_metrics AS (
    SELECT 
        AVG(height) as avg_height,
        AVG(leaf_area) as avg_leaf_area
    FROM tall_plants
)
SELECT * FROM avg_metrics;
```

#### Recursive CTEs
```sql
WITH RECURSIVE plant_growth AS (
    -- Base case
    SELECT 
        plant_id,
        measurement_date,
        height,
        1 as day_number
    FROM plant_measurements
    WHERE measurement_date = '2024-01-01'
    
    UNION ALL
    
    -- Recursive case
    SELECT 
        pm.plant_id,
        pm.measurement_date,
        pm.height,
        pg.day_number + 1
    FROM plant_measurements pm
    INNER JOIN plant_growth pg ON pm.plant_id = pg.plant_id
    WHERE pm.measurement_date = pg.measurement_date + INTERVAL '1 day'
)
SELECT * FROM plant_growth;
```

## 2. Advanced SQL Techniques

### Pivoting Data
```sql
-- Pivot (PostgreSQL)
SELECT 
    plant_id,
    AVG(CASE WHEN measurement_type = 'height' THEN value END) as avg_height,
    AVG(CASE WHEN measurement_type = 'leaf_area' THEN value END) as avg_leaf_area,
    AVG(CASE WHEN measurement_type = 'width' THEN value END) as avg_width
FROM measurements
GROUP BY plant_id;

-- Using PIVOT (SQL Server, Oracle)
SELECT * FROM (
    SELECT plant_id, measurement_type, value
    FROM measurements
) AS source
PIVOT (
    AVG(value)
    FOR measurement_type IN ([height], [leaf_area], [width])
) AS pivoted;
```

### Unpivoting Data
```sql
-- Unpivot
SELECT 
    plant_id,
    'height' as measurement_type,
    height as value
FROM plant_measurements
UNION ALL
SELECT 
    plant_id,
    'leaf_area' as measurement_type,
    leaf_area as value
FROM plant_measurements;
```

### Date/Time Functions
```sql
-- Date operations
SELECT 
    plant_id,
    measurement_date,
    EXTRACT(YEAR FROM measurement_date) as year,
    EXTRACT(MONTH FROM measurement_date) as month,
    EXTRACT(DAY FROM measurement_date) as day,
    DATE_TRUNC('month', measurement_date) as month_start,
    measurement_date + INTERVAL '7 days' as next_week
FROM plant_measurements;

-- Date filtering
SELECT *
FROM plant_measurements
WHERE measurement_date >= CURRENT_DATE - INTERVAL '30 days'
    AND measurement_date < CURRENT_DATE;
```

### String Functions
```sql
-- String operations
SELECT 
    plant_id,
    UPPER(disease_status) as status_upper,
    LOWER(disease_status) as status_lower,
    LENGTH(disease_status) as status_length,
    SUBSTRING(plant_id, 1, 3) as plant_prefix,
    CONCAT(plant_id, '_', disease_status) as plant_status,
    REPLACE(disease_status, ' ', '_') as status_normalized
FROM plant_measurements;

-- Pattern matching
SELECT *
FROM plant_measurements
WHERE disease_status LIKE '%healthy%'
    OR disease_status SIMILAR TO '%(healthy|diseased)%';
```

## 3. PL/SQL Programming

### PL/SQL Basics
```sql
-- PL/SQL block structure
DECLARE
    v_plant_id VARCHAR2(50);
    v_height NUMBER;
    v_avg_height NUMBER;
BEGIN
    -- Get plant height
    SELECT height INTO v_height
    FROM plant_measurements
    WHERE plant_id = 'PLANT001';
    
    -- Calculate average
    SELECT AVG(height) INTO v_avg_height
    FROM plant_measurements;
    
    -- Conditional logic
    IF v_height > v_avg_height THEN
        DBMS_OUTPUT.PUT_LINE('Plant is above average height');
    ELSE
        DBMS_OUTPUT.PUT_LINE('Plant is below average height');
    END IF;
END;
/
```

### Stored Procedures
```sql
-- Create stored procedure
CREATE OR REPLACE PROCEDURE calculate_plant_stats(
    p_plant_id IN VARCHAR2,
    p_avg_height OUT NUMBER,
    p_total_leaf_area OUT NUMBER,
    p_disease_count OUT NUMBER
) AS
BEGIN
    -- Calculate average height
    SELECT AVG(height) INTO p_avg_height
    FROM plant_measurements
    WHERE plant_id = p_plant_id;
    
    -- Calculate total leaf area
    SELECT SUM(leaf_area) INTO p_total_leaf_area
    FROM plant_measurements
    WHERE plant_id = p_plant_id;
    
    -- Count disease occurrences
    SELECT COUNT(*) INTO p_disease_count
    FROM plant_measurements
    WHERE plant_id = p_plant_id
        AND disease_status != 'healthy';
END;
/

-- Call procedure
DECLARE
    v_avg_height NUMBER;
    v_total_leaf_area NUMBER;
    v_disease_count NUMBER;
BEGIN
    calculate_plant_stats(
        'PLANT001',
        v_avg_height,
        v_total_leaf_area,
        v_disease_count
    );
    DBMS_OUTPUT.PUT_LINE('Avg Height: ' || v_avg_height);
    DBMS_OUTPUT.PUT_LINE('Total Leaf Area: ' || v_total_leaf_area);
    DBMS_OUTPUT.PUT_LINE('Disease Count: ' || v_disease_count);
END;
/
```

### Functions
```sql
-- Create function
CREATE OR REPLACE FUNCTION calculate_biomass_estimate(
    p_height IN NUMBER,
    p_leaf_area IN NUMBER
) RETURN NUMBER IS
    v_biomass NUMBER;
BEGIN
    v_biomass := (p_height * 0.5) + (p_leaf_area * 0.3);
    RETURN v_biomass;
END;
/

-- Use function in SQL
SELECT 
    plant_id,
    height,
    leaf_area,
    calculate_biomass_estimate(height, leaf_area) as estimated_biomass
FROM plant_measurements;
```

### Cursors
```sql
-- Explicit cursor
DECLARE
    CURSOR c_plants IS
        SELECT plant_id, height, leaf_area
        FROM plant_measurements
        WHERE height > 10;
    
    v_plant_id VARCHAR2(50);
    v_height NUMBER;
    v_leaf_area NUMBER;
BEGIN
    OPEN c_plants;
    LOOP
        FETCH c_plants INTO v_plant_id, v_height, v_leaf_area;
        EXIT WHEN c_plants%NOTFOUND;
        
        -- Process each row
        DBMS_OUTPUT.PUT_LINE(
            'Plant: ' || v_plant_id || 
            ', Height: ' || v_height
        );
    END LOOP;
    CLOSE c_plants;
END;
/

-- Cursor FOR loop
BEGIN
    FOR rec IN (
        SELECT plant_id, height, leaf_area
        FROM plant_measurements
        WHERE height > 10
    ) LOOP
        DBMS_OUTPUT.PUT_LINE('Plant: ' || rec.plant_id);
    END LOOP;
END;
/
```

### Exception Handling
```sql
CREATE OR REPLACE PROCEDURE safe_update_height(
    p_plant_id IN VARCHAR2,
    p_new_height IN NUMBER
) AS
    v_current_height NUMBER;
BEGIN
    -- Get current height
    SELECT height INTO v_current_height
    FROM plant_measurements
    WHERE plant_id = p_plant_id;
    
    -- Validate
    IF p_new_height < 0 THEN
        RAISE_APPLICATION_ERROR(-20001, 'Height cannot be negative');
    END IF;
    
    -- Update
    UPDATE plant_measurements
    SET height = p_new_height
    WHERE plant_id = p_plant_id;
    
    COMMIT;
    
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Plant not found: ' || p_plant_id);
        RAISE;
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
        ROLLBACK;
        RAISE;
END;
/
```

## 4. Snowflake Data Warehouse

### Snowflake Overview
- Cloud-native data warehouse
- Automatic scaling
- Separation of storage and compute
- Time travel and cloning
- Zero-copy cloning

### Snowflake SQL
```sql
-- Create database
CREATE DATABASE plant_phenotyping_db;

-- Create schema
CREATE SCHEMA raw_data;
CREATE SCHEMA processed_data;
CREATE SCHEMA ml_features;

-- Create table
CREATE TABLE plant_measurements (
    plant_id VARCHAR(50),
    measurement_date DATE,
    height FLOAT,
    leaf_area FLOAT,
    disease_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Load data from stage
CREATE STAGE plant_data_stage
    URL = 's3://bucket/plant-data/'
    CREDENTIALS = (AWS_KEY_ID='...' AWS_SECRET_KEY='...');

COPY INTO plant_measurements
FROM @plant_data_stage
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1);
```

### Snowflake Features

#### Clustering
```sql
-- Cluster table for better performance
ALTER TABLE plant_measurements
CLUSTER BY (plant_id, measurement_date);
```

#### Time Travel
```sql
-- Query historical data
SELECT * FROM plant_measurements
AT (TIMESTAMP => '2024-01-01 10:00:00'::TIMESTAMP);

-- Query before drop
SELECT * FROM plant_measurements
BEFORE (STATEMENT => '01a2b3c4-5678-90ab-cdef-1234567890ab');
```

#### Zero-Copy Cloning
```sql
-- Clone table
CREATE TABLE plant_measurements_dev CLONE plant_measurements;

-- Clone database
CREATE DATABASE plant_phenotyping_dev CLONE plant_phenotyping_db;
```

#### Materialized Views
```sql
-- Create materialized view
CREATE MATERIALIZED VIEW plant_daily_stats AS
SELECT 
    DATE_TRUNC('day', measurement_date) as date,
    disease_status,
    AVG(height) as avg_height,
    AVG(leaf_area) as avg_leaf_area,
    COUNT(*) as measurement_count
FROM plant_measurements
GROUP BY DATE_TRUNC('day', measurement_date), disease_status;

-- Refresh materialized view
ALTER MATERIALIZED VIEW plant_daily_stats REFRESH;
```

### Snowflake Python Connector
```python
import snowflake.connector
import pandas as pd

# Connect to Snowflake
conn = snowflake.connector.connect(
    user='username',
    password='password',
    account='account',
    warehouse='COMPUTE_WH',
    database='plant_phenotyping_db',
    schema='raw_data'
)

# Execute query
cursor = conn.cursor()
cursor.execute("SELECT * FROM plant_measurements LIMIT 1000")
df = cursor.fetch_pandas_all()

# Using SQLAlchemy
from sqlalchemy import create_engine

engine = create_engine(
    'snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}'
)

df = pd.read_sql("SELECT * FROM plant_measurements", engine)

# Write to Snowflake
df.to_sql('plant_measurements', engine, if_exists='append', index=False)
```

## 5. Database Design for ML

### Schema Design
```sql
-- Fact table (measurements)
CREATE TABLE fact_plant_measurements (
    measurement_id BIGINT PRIMARY KEY,
    plant_id VARCHAR(50),
    measurement_date DATE,
    height FLOAT,
    leaf_area FLOAT,
    width FLOAT,
    disease_status VARCHAR(20),
    measurement_type VARCHAR(20),
    created_at TIMESTAMP
);

-- Dimension tables
CREATE TABLE dim_plants (
    plant_id VARCHAR(50) PRIMARY KEY,
    plant_species VARCHAR(50),
    planting_date DATE,
    location VARCHAR(100),
    treatment_type VARCHAR(50)
);

CREATE TABLE dim_diseases (
    disease_id INT PRIMARY KEY,
    disease_name VARCHAR(50),
    disease_category VARCHAR(50),
    severity_level VARCHAR(20)
);

-- Star schema query
SELECT 
    p.plant_species,
    d.disease_name,
    DATE_TRUNC('month', m.measurement_date) as month,
    AVG(m.height) as avg_height,
    COUNT(*) as measurement_count
FROM fact_plant_measurements m
JOIN dim_plants p ON m.plant_id = p.plant_id
JOIN dim_diseases d ON m.disease_status = d.disease_name
GROUP BY p.plant_species, d.disease_name, DATE_TRUNC('month', m.measurement_date);
```

### Indexing Strategy
```sql
-- Create indexes for common queries
CREATE INDEX idx_plant_date ON plant_measurements(plant_id, measurement_date);
CREATE INDEX idx_disease_status ON plant_measurements(disease_status);
CREATE INDEX idx_height ON plant_measurements(height) WHERE height > 10;

-- Analyze table for query optimization
ANALYZE TABLE plant_measurements;
```

## 6. ETL with SQL

### Incremental Load
```sql
-- Incremental load procedure
CREATE OR REPLACE PROCEDURE incremental_load_plants AS
    v_last_load_date DATE;
BEGIN
    -- Get last load date
    SELECT MAX(measurement_date) INTO v_last_load_date
    FROM plant_measurements;
    
    -- Insert new records
    INSERT INTO plant_measurements
    SELECT *
    FROM staging_plant_measurements
    WHERE measurement_date > v_last_load_date;
    
    COMMIT;
END;
/
```

### Data Transformation
```sql
-- Transform and load
INSERT INTO processed_plant_measurements
SELECT 
    plant_id,
    measurement_date,
    height,
    leaf_area,
    CASE 
        WHEN height < 10 THEN 'short'
        WHEN height < 20 THEN 'medium'
        ELSE 'tall'
    END as height_category,
    CASE
        WHEN disease_status IN ('healthy', 'normal') THEN 'healthy'
        ELSE 'diseased'
    END as health_status,
    height * 0.5 + leaf_area * 0.3 as estimated_biomass
FROM raw_plant_measurements
WHERE height > 0 AND leaf_area > 0;
```

## 7. Performance Optimization

### Query Optimization
```sql
-- Use EXPLAIN PLAN
EXPLAIN PLAN FOR
SELECT * FROM plant_measurements
WHERE plant_id = 'PLANT001'
    AND measurement_date >= '2024-01-01';

-- Analyze execution plan
SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
```

### Best Practices
1. **Use indexes** on frequently queried columns
2. **Partition large tables** by date or category
3. **Avoid SELECT *** - specify needed columns
4. **Use WHERE clauses** to filter early
5. **Use appropriate joins** (INNER vs LEFT)
6. **Limit result sets** when possible
7. **Use EXPLAIN** to understand query plans
8. **Update statistics** regularly

## 8. Integration with ML Pipelines

### Python Integration
```python
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    database="plant_db",
    user="username",
    password="password"
)

# Read data
df = pd.read_sql("""
    SELECT 
        plant_id,
        height,
        leaf_area,
        disease_status
    FROM plant_measurements
    WHERE measurement_date >= CURRENT_DATE - INTERVAL '30 days'
""", conn)

# Write data
df.to_sql('ml_predictions', conn, if_exists='append', index=False)

# Execute stored procedure
cursor = conn.cursor()
cursor.callproc('calculate_plant_stats', ['PLANT001'])
results = cursor.fetchall()
```

## 9. Best Practices

1. **Normalize data** to reduce redundancy
2. **Use transactions** for data integrity
3. **Implement constraints** (PRIMARY KEY, FOREIGN KEY, CHECK)
4. **Regular backups** and recovery testing
5. **Monitor performance** and optimize queries
6. **Document schemas** and procedures
7. **Use connection pooling** for applications
8. **Secure access** with proper authentication

## Next Steps

Continue to [Module 8: Cloud Deployment](08-cloud-deployment.md) to learn about deploying ML models on Azure and GCP.

