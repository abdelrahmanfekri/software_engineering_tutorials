# Statistics Tutorial 01: Descriptive Statistics

## Learning Objectives
By the end of this tutorial, you will be able to:
- Calculate measures of central tendency
- Compute measures of variability
- Understand measures of shape
- Create and interpret graphical representations
- Summarize data effectively
- Choose appropriate summary statistics

## Introduction to Descriptive Statistics

### What is Descriptive Statistics?
Descriptive statistics summarize and describe the main features of a dataset. They provide a way to understand data without making inferences about a larger population.

### Types of Data
1. **Quantitative**: Numerical data (continuous or discrete)
2. **Qualitative**: Categorical data (nominal or ordinal)

**Examples**:
- Quantitative: Height, weight, test scores
- Qualitative: Gender, color, rating scale

## Measures of Central Tendency

### Mean (Arithmetic Average)
x̄ = Σx/n

**Example**: Data: 2, 4, 6, 8, 10
x̄ = (2 + 4 + 6 + 8 + 10)/5 = 30/5 = 6

### Median
The middle value when data is ordered.

**Example**: Data: 1, 3, 5, 7, 9
Median = 5 (middle value)

**Example**: Data: 1, 3, 5, 7, 9, 11
Median = (5 + 7)/2 = 6 (average of two middle values)

### Mode
The most frequently occurring value.

**Example**: Data: 2, 3, 3, 4, 5, 5, 5
Mode = 5 (appears 3 times)

### When to Use Each Measure
- **Mean**: Best for symmetric data without outliers
- **Median**: Best for skewed data or when outliers are present
- **Mode**: Best for categorical data or when identifying most common value

## Measures of Variability

### Range
Range = Maximum - Minimum

**Example**: Data: 2, 5, 8, 12, 15
Range = 15 - 2 = 13

### Variance
Population variance: σ² = Σ(x - μ)²/N
Sample variance: s² = Σ(x - x̄)²/(n - 1)

**Example**: Data: 2, 4, 6, 8, 10 (sample)
x̄ = 6
s² = [(2-6)² + (4-6)² + (6-6)² + (8-6)² + (10-6)²]/(5-1)
s² = [16 + 4 + 0 + 4 + 16]/4 = 40/4 = 10

### Standard Deviation
s = √s²

**Example**: From above, s = √10 ≈ 3.16

### Interquartile Range (IQR)
IQR = Q₃ - Q₁

Where Q₁ (first quartile) and Q₃ (third quartile) divide data into quarters.

**Example**: Data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
Q₁ = 3, Q₃ = 8
IQR = 8 - 3 = 5

## Measures of Shape

### Skewness
Measures asymmetry of distribution:
- **Positive skew**: Tail extends to the right
- **Negative skew**: Tail extends to the left
- **Symmetric**: Bell-shaped curve

### Kurtosis
Measures "peakedness" of distribution:
- **Leptokurtic**: More peaked than normal
- **Mesokurtic**: Normal peakedness
- **Platykurtic**: Less peaked than normal

## Graphical Representations

### Histogram
Shows frequency distribution of continuous data.

**Steps**:
1. Choose number of bins
2. Calculate bin width
3. Count observations in each bin
4. Draw bars with heights proportional to frequencies

### Box Plot (Box-and-Whisker Plot)
Shows five-number summary: minimum, Q₁, median, Q₃, maximum.

**Components**:
- Box: Q₁ to Q₃ (IQR)
- Line in box: Median
- Whiskers: Extend to 1.5×IQR or data limits
- Outliers: Points beyond whiskers

### Scatter Plot
Shows relationship between two quantitative variables.

**Example**: Height vs. Weight
- X-axis: Height
- Y-axis: Weight
- Each point represents one person

### Bar Chart
Shows frequencies of categorical data.

**Example**: Number of students by major
- X-axis: Major categories
- Y-axis: Number of students
- Bars represent counts

## Summary Statistics

### Five-Number Summary
1. Minimum
2. First quartile (Q₁)
3. Median (Q₂)
4. Third quartile (Q₃)
5. Maximum

### Mean and Standard Deviation
- Mean: Center of distribution
- Standard deviation: Spread of distribution

### Coefficient of Variation
CV = (s/x̄) × 100%

Measures relative variability.

**Example**: If x̄ = 50 and s = 10, then CV = (10/50) × 100% = 20%

## Data Analysis Process

### 1. Data Collection
- Ensure data quality
- Check for missing values
- Verify data types

### 2. Exploratory Data Analysis
- Calculate summary statistics
- Create visualizations
- Identify patterns and outliers

### 3. Data Cleaning
- Handle missing values
- Remove or correct outliers
- Standardize formats

### 4. Analysis and Interpretation
- Choose appropriate statistics
- Create meaningful visualizations
- Draw conclusions

## Practice Problems

### Problem 1
Calculate mean, median, and mode for: 3, 7, 2, 9, 5, 7, 1

**Solution**:
- Mean: (3 + 7 + 2 + 9 + 5 + 7 + 1)/7 = 34/7 ≈ 4.86
- Median: Ordered data: 1, 2, 3, 5, 7, 7, 9 → Median = 5
- Mode: 7 (appears twice)

### Problem 2
Find range, variance, and standard deviation for: 10, 12, 14, 16, 18

**Solution**:
- Range: 18 - 10 = 8
- Mean: (10 + 12 + 14 + 16 + 18)/5 = 14
- Variance: [(10-14)² + (12-14)² + (14-14)² + (16-14)² + (18-14)²]/4
- Variance: [16 + 4 + 0 + 4 + 16]/4 = 40/4 = 10
- Standard deviation: √10 ≈ 3.16

### Problem 3
Create a box plot for: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20

**Solution**:
- Q₁ = 6, Median = 11, Q₃ = 16
- IQR = 16 - 6 = 10
- Lower whisker: 6 - 1.5(10) = -9, but data minimum is 2
- Upper whisker: 16 + 1.5(10) = 31, but data maximum is 20
- Box plot shows median at 11, box from 6 to 16, whiskers to 2 and 20

## Key Takeaways
- Descriptive statistics summarize data characteristics
- Mean, median, and mode measure central tendency
- Range, variance, and standard deviation measure variability
- Skewness and kurtosis describe distribution shape
- Graphical representations provide visual insights
- Choose statistics appropriate for data type and distribution

## Next Steps
In the next tutorial, we'll explore probability distributions, learning about normal distribution and other important distributions used in statistical analysis.
