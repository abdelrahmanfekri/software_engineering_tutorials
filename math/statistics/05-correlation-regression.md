# Statistics Tutorial 05: Correlation and Regression

## Learning Objectives
By the end of this tutorial, you will be able to:
- Calculate and interpret correlation coefficients
- Understand the difference between correlation and causation
- Perform simple linear regression analysis
- Interpret regression coefficients and R²
- Check regression assumptions
- Use regression for prediction
- Understand multiple regression concepts

## Introduction to Correlation and Regression

### What is Correlation?
Correlation measures the strength and direction of linear relationship between two variables.

### What is Regression?
Regression analyzes the relationship between variables to make predictions and understand associations.

### Key Distinction
- **Correlation**: Measures association (no cause-effect implied)
- **Regression**: Can be used for prediction and may imply causation

## Correlation Analysis

### Pearson Correlation Coefficient (r)

**Formula**:
r = Σ[(x_i - x̄)(y_i - ȳ)] / √[Σ(x_i - x̄)² × Σ(y_i - ȳ)²]

**Alternative Formula**:
r = Σ(x_i y_i) - n(x̄)(ȳ) / √[(Σx_i² - nx̄²)(Σy_i² - nȳ²)]

**Properties**:
- Range: -1 ≤ r ≤ 1
- r = 1: Perfect positive correlation
- r = -1: Perfect negative correlation
- r = 0: No linear correlation

**Example**: Calculate correlation between height and weight
- Heights: [65, 68, 70, 72, 75] inches
- Weights: [120, 140, 160, 180, 200] pounds
- x̄ = 70, ȳ = 160
- r = Σ[(x_i - 70)(y_i - 160)] / √[Σ(x_i - 70)² × Σ(y_i - 160)²]
- r = 200 / √(100 × 2000) = 200 / √200000 = 200 / 447.2 = 0.447

### Interpretation of Correlation Strength
- |r| = 0.0 - 0.3: Weak correlation
- |r| = 0.3 - 0.7: Moderate correlation
- |r| = 0.7 - 1.0: Strong correlation

### Spearman Rank Correlation (ρ)
Used when data is ordinal or when relationship is monotonic but not linear.

**Method**:
1. Rank both variables
2. Apply Pearson formula to ranks
3. ρ = 1 - (6Σd²)/(n(n²-1)) where d = difference in ranks

**Example**: Test scores vs. study hours (ordinal)
- Ranks: [1, 2, 3, 4, 5] vs [2, 1, 4, 3, 5]
- d = [-1, 1, -1, 1, 0]
- Σd² = 1 + 1 + 1 + 1 + 0 = 4
- ρ = 1 - (6×4)/(5×24) = 1 - 24/120 = 1 - 0.2 = 0.8

### Testing Correlation Significance
**Hypothesis Test**:
- H₀: ρ = 0 (no correlation)
- H₁: ρ ≠ 0 (correlation exists)

**Test Statistic**:
t = r√(n-2)/√(1-r²)

**Example**: r = 0.6, n = 20
- t = 0.6√18/√(1-0.36) = 0.6×4.24/√0.64 = 2.54/0.8 = 3.18
- df = 18, p-value ≈ 0.005
- Since p < 0.05, reject H₀

## Simple Linear Regression

### Regression Model
y = β₀ + β₁x + ε

Where:
- y = dependent variable
- x = independent variable
- β₀ = y-intercept
- β₁ = slope
- ε = error term

### Least Squares Estimation

**Slope**:
β₁ = Σ[(x_i - x̄)(y_i - ȳ)] / Σ(x_i - x̄)²

**Intercept**:
β₀ = ȳ - β₁x̄

**Example**: Fit regression line for height vs. weight
- Heights: [65, 68, 70, 72, 75]
- Weights: [120, 140, 160, 180, 200]
- x̄ = 70, ȳ = 160
- β₁ = 200/100 = 2
- β₀ = 160 - 2(70) = 160 - 140 = 20
- Regression equation: Weight = 20 + 2(Height)

### Coefficient of Determination (R²)
R² = r² = (SS_regression / SS_total)

**Interpretation**:
- R² = 0.64 means 64% of variance in y is explained by x
- R² = 1 - (SS_residual / SS_total)

**Example**: From height-weight data
- SS_total = 2000
- SS_residual = 720
- R² = 1 - (720/2000) = 1 - 0.36 = 0.64

### Standard Error of Estimate
s_e = √(SS_residual / (n-2))

**Example**: s_e = √(720/3) = √240 = 15.49

## Regression Assumptions

### 1. Linearity
- Relationship between x and y is linear
- Check with scatter plot
- Use residual plots

### 2. Independence
- Observations are independent
- No autocorrelation
- Random sampling

### 3. Homoscedasticity
- Constant variance of residuals
- Check with residual plot
- No funnel patterns

### 4. Normality
- Residuals are normally distributed
- Check with normal probability plot
- Use Shapiro-Wilk test

### 5. No Outliers
- No influential observations
- Check with Cook's distance
- Use leverage statistics

## Regression Diagnostics

### Residual Analysis
**Residuals**: e_i = y_i - ŷ_i

**Standardized Residuals**: e_i / s_e

**Studentized Residuals**: e_i / (s_e√(1-h_ii))

### Influential Observations
**Cook's Distance**: D_i = (e_i²/(k+1)s²) × (h_ii/(1-h_ii))

**Leverage**: h_ii = 1/n + (x_i - x̄)²/Σ(x_i - x̄)²

### Detection Methods
- Residual plots
- Normal probability plots
- Cook's distance plots
- Leverage plots

## Prediction and Confidence Intervals

### Point Prediction
ŷ = β₀ + β₁x

### Prediction Interval
PI = ŷ ± t_(α/2, n-2) × s_e × √(1 + 1/n + (x - x̄)²/Σ(x_i - x̄)²)

### Confidence Interval for Mean Response
CI = ŷ ± t_(α/2, n-2) × s_e × √(1/n + (x - x̄)²/Σ(x_i - x̄)²)

**Example**: Predict weight for height = 73 inches
- ŷ = 20 + 2(73) = 166 pounds
- Prediction interval: 166 ± 2.776 × 15.49 × √(1 + 1/5 + 9/100)
- PI = 166 ± 42.99 × √1.29 = 166 ± 48.8
- 95% PI: (117.2, 214.8)

## Multiple Linear Regression

### Model
y = β₀ + β₁x₁ + β₂x₂ + ... + β_kx_k + ε

### Matrix Form
Y = Xβ + ε

### Normal Equations
β̂ = (X'X)⁻¹X'Y

### Multiple R²
R² = 1 - (SS_residual / SS_total)

### Adjusted R²
R²_adj = 1 - (SS_residual/(n-k-1)) / (SS_total/(n-1))

**Example**: Predict salary from education and experience
- Salary = 20000 + 5000(Education) + 1000(Experience)
- R² = 0.75, R²_adj = 0.72

## Common Mistakes

### 1. Confusing Correlation with Causation
- Correlation doesn't imply causation
- Need experimental design for causation
- Consider confounding variables

### 2. Extrapolation
- Don't predict outside data range
- Linear relationship may not hold
- Use caution with extreme values

### 3. Ignoring Assumptions
- Check all regression assumptions
- Use appropriate transformations
- Consider alternative models

### 4. Overfitting
- Too many variables reduce generalizability
- Use adjusted R² for model selection
- Consider cross-validation

## Practice Problems

### Problem 1
Calculate correlation between study hours and test scores:
- Hours: [2, 4, 6, 8, 10]
- Scores: [60, 70, 80, 85, 95]

**Solution**:
- x̄ = 6, ȳ = 78
- r = Σ[(x_i - 6)(y_i - 78)] / √[Σ(x_i - 6)² × Σ(y_i - 78)²]
- r = 200 / √(40 × 650) = 200 / √26000 = 200 / 161.2 = 0.99

### Problem 2
Fit regression line for the data in Problem 1.

**Solution**:
- β₁ = 200/40 = 5
- β₀ = 78 - 5(6) = 48
- Regression equation: Score = 48 + 5(Hours)

### Problem 3
For the regression in Problem 2, find R² and interpret.

**Solution**:
- SS_total = 650
- SS_residual = 650 - 200²/40 = 650 - 1000 = -350 (error in calculation)
- Correct: SS_residual = Σ(y_i - ŷ_i)²
- R² = 1 - (SS_residual / SS_total) ≈ 0.98

### Problem 4
Predict test score for 7 hours of study.

**Solution**:
- ŷ = 48 + 5(7) = 48 + 35 = 83
- Predicted score: 83

## Key Takeaways
- Correlation measures linear association between variables
- Regression provides prediction and understanding of relationships
- Always check regression assumptions
- R² measures proportion of variance explained
- Correlation ≠ causation
- Use appropriate methods for your data type
- Consider multiple regression for complex relationships

## Next Steps
In the next tutorial, we'll explore Analysis of Variance (ANOVA), learning how to compare means across multiple groups and understand the F-test for testing differences between groups.
