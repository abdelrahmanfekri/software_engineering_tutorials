# Statistics Tutorial 08: Advanced Topics

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand multivariate statistical methods
- Perform principal component analysis (PCA)
- Conduct factor analysis
- Apply cluster analysis techniques
- Understand time series analysis concepts
- Learn Bayesian statistical methods
- Apply advanced regression techniques
- Use machine learning statistics

## Introduction to Advanced Statistics

### What are Advanced Statistical Methods?
Advanced statistical methods extend basic statistical concepts to handle complex data structures, multiple variables, and sophisticated modeling approaches.

### When to Use Advanced Methods
1. **Multiple variables** to analyze simultaneously
2. **Complex relationships** between variables
3. **Time-dependent data**
4. **High-dimensional data**
5. **Uncertainty quantification**
6. **Predictive modeling**

## Multivariate Statistics

### Multiple Regression Analysis

#### Multiple Linear Regression
**Model**: y = β₀ + β₁x₁ + β₂x₂ + ... + β_kx_k + ε

**Matrix Form**: Y = Xβ + ε

**Estimation**: β̂ = (X'X)⁻¹X'Y

#### Example: Predict salary from education and experience
- Salary = 20000 + 5000(Education) + 1000(Experience)
- R² = 0.75, Adjusted R² = 0.72
- Both predictors significant (p < 0.05)

#### Assumptions
1. **Linearity**: Linear relationship between predictors and response
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Predictors not highly correlated

#### Model Selection
- **Forward Selection**: Add variables one by one
- **Backward Elimination**: Remove variables one by one
- **Stepwise**: Combine forward and backward
- **Information Criteria**: AIC, BIC for model comparison

### Principal Component Analysis (PCA)

#### Purpose
Reduce dimensionality while preserving maximum variance.

#### Method
1. **Standardize variables**
2. **Calculate correlation matrix**
3. **Find eigenvalues and eigenvectors**
4. **Select principal components**

#### Example: Reduce 4 variables to 2 components
- Variables: Height, Weight, Age, Income
- Correlation matrix: 4×4 matrix
- Eigenvalues: [2.1, 1.2, 0.5, 0.2]
- First two components explain 82.5% of variance

#### Interpretation
- **Loadings**: Correlation between variables and components
- **Scores**: Component values for each observation
- **Variance explained**: Proportion of total variance

### Factor Analysis

#### Purpose
Identify underlying latent factors that explain correlations among observed variables.

#### Model
X = ΛF + ε

Where:
- X = observed variables
- Λ = factor loadings
- F = latent factors
- ε = unique factors

#### Example: Intelligence test analysis
- Variables: Math, Verbal, Spatial, Memory scores
- Factors: General Intelligence, Specific Abilities
- Loadings show relationship between tests and factors

#### Factor Rotation
- **Varimax**: Maximize variance of loadings
- **Promax**: Allow correlation between factors
- **Oblimin**: General oblique rotation

### Cluster Analysis

#### Purpose
Group observations into clusters based on similarity.

#### Methods

##### K-Means Clustering
1. **Choose number of clusters (k)**
2. **Initialize cluster centers**
3. **Assign observations to nearest center**
4. **Update cluster centers**
5. **Repeat until convergence**

##### Hierarchical Clustering
1. **Start with each observation as cluster**
2. **Merge closest clusters**
3. **Repeat until single cluster**
4. **Use dendrogram to choose clusters**

#### Example: Customer segmentation
- Variables: Age, Income, Spending
- 3 clusters: Young High-Spenders, Middle-Age Savers, Older Conservative
- Use cluster centers to characterize segments

## Time Series Analysis

### Components of Time Series
1. **Trend**: Long-term direction
2. **Seasonality**: Regular patterns
3. **Cyclical**: Irregular patterns
4. **Random**: Unexplained variation

### Autocorrelation
**Autocorrelation Function (ACF)**:
ACF(k) = Corr(X_t, X_{t-k})

**Partial Autocorrelation Function (PACF)**:
PACF(k) = Corr(X_t, X_{t-k} | X_{t-1}, ..., X_{t-k+1})

### ARIMA Models
**ARIMA(p,d,q)**:
- **AR(p)**: Autoregressive component
- **I(d)**: Integrated component (differencing)
- **MA(q)**: Moving average component

#### Example: Stock price prediction
- Model: ARIMA(1,1,1)
- AR(1): X_t = φX_{t-1} + ε_t
- I(1): First difference
- MA(1): ε_t = θε_{t-1} + a_t

### Forecasting Methods
1. **Exponential Smoothing**: Weight recent observations more
2. **ARIMA**: Autoregressive integrated moving average
3. **Seasonal Decomposition**: Separate trend and seasonal components
4. **Machine Learning**: Neural networks, random forests

## Bayesian Statistics

### Bayesian vs. Frequentist Approach
- **Frequentist**: Parameters are fixed, data is random
- **Bayesian**: Parameters are random, data is fixed

### Bayes' Theorem
P(θ|data) = P(data|θ) × P(θ) / P(data)

Where:
- P(θ|data) = Posterior distribution
- P(data|θ) = Likelihood
- P(θ) = Prior distribution
- P(data) = Marginal likelihood

### Example: Coin flipping
- Prior: P(heads) ~ Beta(2,2) (weak prior)
- Data: 7 heads out of 10 flips
- Posterior: P(heads) ~ Beta(9,5)
- Posterior mean: 9/(9+5) = 0.64

### Markov Chain Monte Carlo (MCMC)
**Purpose**: Sample from complex posterior distributions

**Methods**:
1. **Metropolis-Hastings**: General purpose sampling
2. **Gibbs Sampling**: Sample from conditional distributions
3. **Hamiltonian Monte Carlo**: More efficient sampling

### Applications
- **Parameter estimation**: Uncertainty quantification
- **Model comparison**: Bayes factors
- **Prediction**: Posterior predictive distributions
- **Decision making**: Optimal decisions under uncertainty

## Machine Learning Statistics

### Cross-Validation
**Purpose**: Assess model performance and prevent overfitting

**Methods**:
1. **K-Fold**: Divide data into k folds
2. **Leave-One-Out**: Use n-1 observations for training
3. **Stratified**: Maintain class proportions
4. **Time Series**: Respect temporal order

### Model Selection Criteria
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion
- **Cross-Validation**: Direct performance estimation
- **Regularization**: Penalize complex models

### Ensemble Methods
1. **Bagging**: Bootstrap aggregating
2. **Boosting**: Sequential model building
3. **Random Forest**: Multiple decision trees
4. **Stacking**: Meta-learning approach

### Overfitting and Underfitting
- **Overfitting**: Model too complex, poor generalization
- **Underfitting**: Model too simple, poor fit
- **Bias-Variance Tradeoff**: Balance between bias and variance

## Advanced Regression Techniques

### Logistic Regression
**Purpose**: Predict binary outcomes

**Model**: logit(p) = β₀ + β₁x₁ + ... + β_kx_k

**Example**: Predict customer churn
- Variables: Age, Income, Usage, Support calls
- Outcome: Churn (0/1)
- Coefficients: Odds ratios

### Poisson Regression
**Purpose**: Predict count data

**Model**: log(λ) = β₀ + β₁x₁ + ... + β_kx_k

**Example**: Predict number of accidents
- Variables: Traffic volume, Weather, Time of day
- Outcome: Number of accidents
- Coefficients: Rate ratios

### Survival Analysis
**Purpose**: Analyze time-to-event data

**Methods**:
1. **Kaplan-Meier**: Nonparametric survival curves
2. **Cox Regression**: Proportional hazards model
3. **Accelerated Failure Time**: Parametric models

### Generalized Linear Models (GLM)
**Framework**: Extends linear regression to non-normal distributions

**Components**:
1. **Random Component**: Distribution of response
2. **Systematic Component**: Linear predictor
3. **Link Function**: Connects mean to linear predictor

## Big Data Statistics

### Challenges
1. **Volume**: Large datasets
2. **Velocity**: Real-time data streams
3. **Variety**: Different data types
4. **Veracity**: Data quality issues

### Solutions
1. **Sampling**: Use representative samples
2. **Distributed Computing**: Parallel processing
3. **Streaming Analytics**: Real-time analysis
4. **Data Quality**: Cleaning and validation

### Tools
- **R**: Statistical computing
- **Python**: pandas, scipy, scikit-learn
- **Spark**: Distributed computing
- **Hadoop**: Big data processing

## Ethical Considerations

### Data Privacy
- **Anonymization**: Remove identifying information
- **Differential Privacy**: Add noise to protect privacy
- **Consent**: Informed consent for data use
- **Transparency**: Clear data usage policies

### Algorithmic Bias
- **Fairness**: Equal treatment across groups
- **Transparency**: Explainable AI
- **Accountability**: Responsibility for decisions
- **Auditing**: Regular bias assessments

### Reproducibility
- **Open Data**: Share datasets
- **Open Code**: Share analysis code
- **Documentation**: Clear methodology
- **Peer Review**: Independent verification

## Practice Problems

### Problem 1
Perform PCA on correlation matrix:
```
     V1   V2   V3   V4
V1  1.0  0.8  0.6  0.4
V2  0.8  1.0  0.7  0.5
V3  0.6  0.7  1.0  0.3
V4  0.4  0.5  0.3  1.0
```

**Solution**:
- Eigenvalues: [2.8, 0.9, 0.2, 0.1]
- First component explains 70% of variance
- Loadings show V1 and V2 contribute most to PC1

### Problem 2
Bayesian analysis with normal prior:
- Prior: μ ~ N(100, 25)
- Data: n = 20, x̄ = 105, s = 10
- Find posterior distribution

**Solution**:
- Posterior: μ ~ N(104.2, 4.8)
- Posterior mean: 104.2
- Posterior variance: 4.8

### Problem 3
Time series decomposition:
- Trend: Linear increase
- Seasonal: Quarterly pattern
- Random: White noise
- Forecast next period

**Solution**:
- Separate components
- Project trend forward
- Add seasonal component
- Include confidence intervals

## Key Takeaways
- Advanced methods handle complex data structures
- Multivariate methods analyze multiple variables simultaneously
- Time series methods handle temporal data
- Bayesian methods incorporate prior information
- Machine learning methods provide predictive power
- Consider ethical implications of advanced methods
- Choose methods appropriate for your data and goals

## Machine Learning Statistics

### Cross-Validation and Model Selection
Cross-validation is essential for evaluating machine learning models and preventing overfitting.

#### K-Fold Cross-Validation
1. **Split data** into k equal parts
2. **Train model** on k-1 parts
3. **Test model** on remaining part
4. **Repeat** for all k folds
5. **Average** performance across folds

**Example**: 5-fold CV for linear regression
- Fold 1: Train on parts 2,3,4,5; test on part 1
- Fold 2: Train on parts 1,3,4,5; test on part 2
- ... (continue for all folds)
- Average R² across all folds

#### Leave-One-Out Cross-Validation (LOOCV)
Special case where k = n (number of samples).

**Advantages**:
- Uses maximum data for training
- Unbiased estimate of performance

**Disadvantages**:
- Computationally expensive
- High variance in estimates

### Bootstrap Methods
Bootstrap provides non-parametric estimates of confidence intervals and standard errors.

#### Bootstrap Algorithm
1. **Sample with replacement** from original data (n samples)
2. **Compute statistic** on bootstrap sample
3. **Repeat** B times (typically B = 1000)
4. **Estimate distribution** of statistic

**Example**: Bootstrap confidence interval for mean
```python
import numpy as np

def bootstrap_mean(data, B=1000, alpha=0.05):
    n = len(data)
    bootstrap_means = []
    
    for _ in range(B):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    lower = np.percentile(bootstrap_means, 100*alpha/2)
    upper = np.percentile(bootstrap_means, 100*(1-alpha/2))
    
    return lower, upper
```

### Information Criteria
Information criteria balance model fit with model complexity.

#### Akaike Information Criterion (AIC)
AIC = 2k - 2ln(L)

Where:
- k = number of parameters
- L = likelihood of the model

#### Bayesian Information Criterion (BIC)
BIC = k·ln(n) - 2ln(L)

Where:
- n = sample size
- k = number of parameters
- L = likelihood of the model

**Model Selection**: Choose model with lowest AIC/BIC

### Regularization Methods
Regularization prevents overfitting by adding penalty terms to the loss function.

#### Ridge Regression (L2 Regularization)
**Objective**: Minimize ||y - Xβ||² + λ||β||²

**Solution**: β̂ = (X^T X + λI)^(-1) X^T y

**Properties**:
- Shrinks coefficients toward zero
- Handles multicollinearity
- Always has unique solution

#### Lasso Regression (L1 Regularization)
**Objective**: Minimize ||y - Xβ||² + λ||β||₁

**Properties**:
- Can set coefficients to exactly zero
- Performs automatic feature selection
- May not have unique solution

#### Elastic Net
**Objective**: Minimize ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²

Combines benefits of both Ridge and Lasso.

### Bayesian Machine Learning
Bayesian methods provide uncertainty quantification in machine learning.

#### Bayesian Linear Regression
**Prior**: β ~ N(0, α^(-1)I)
**Likelihood**: y|X,β ~ N(Xβ, β^(-1)I)
**Posterior**: β|y,X ~ N(μ_N, S_N)

Where:
- μ_N = βS_N X^T y
- S_N^(-1) = αI + βX^T X

#### Variational Inference
Approximate intractable posterior distributions using optimization.

**Evidence Lower Bound (ELBO)**:
ELBO = E_q[log p(x,z)] - E_q[log q(z)]

Maximizing ELBO minimizes KL divergence between approximate and true posterior.

### Statistical Learning Theory
Theoretical foundations for understanding machine learning performance.

#### PAC Learning
**Probably Approximately Correct (PAC)** learning framework:

**Definition**: A concept class C is PAC-learnable if there exists an algorithm that, given ε, δ > 0 and sufficient samples, outputs a hypothesis h such that:
- P[error(h) ≤ ε] ≥ 1 - δ

#### VC Dimension
**Vapnik-Chervonenkis dimension** measures model complexity.

**Definition**: VC dimension is the largest number of points that can be shattered by the hypothesis class.

**Example**: Linear classifiers in 2D have VC dimension = 3

#### Generalization Bounds
**Theorem**: With probability 1-δ, for any hypothesis h:
error(h) ≤ error_S(h) + √((VC(H) + log(1/δ))/(2n))

Where:
- error(h) = true error
- error_S(h) = training error
- VC(H) = VC dimension
- n = sample size

### Ensemble Methods
Ensemble methods combine multiple models to improve performance.

#### Bagging (Bootstrap Aggregating)
1. **Train multiple models** on bootstrap samples
2. **Average predictions** for regression
3. **Vote** for classification

**Example**: Random Forest
- Multiple decision trees
- Each tree trained on bootstrap sample
- Random feature selection at each split

#### Boosting
1. **Train models sequentially**
2. **Weight misclassified samples** more heavily
3. **Combine models** with learned weights

**Example**: AdaBoost
- Start with equal weights
- Train weak learner
- Increase weights of misclassified samples
- Repeat with updated weights

#### Stacking
1. **Train base models** on training data
2. **Train meta-model** on base model predictions
3. **Use meta-model** for final predictions

### Model Evaluation Metrics

#### Classification Metrics
- **Accuracy**: (TP + TN)/(TP + TN + FP + FN)
- **Precision**: TP/(TP + FP)
- **Recall**: TP/(TP + FN)
- **F1-Score**: 2·Precision·Recall/(Precision + Recall)
- **ROC-AUC**: Area under ROC curve

#### Regression Metrics
- **Mean Squared Error (MSE)**: (1/n)Σ(y_i - ŷ_i)²
- **Root Mean Squared Error (RMSE)**: √MSE
- **Mean Absolute Error (MAE)**: (1/n)Σ|y_i - ŷ_i|
- **R²**: 1 - SS_res/SS_tot

### Hyperparameter Optimization
Systematic approaches to finding optimal hyperparameters.

#### Grid Search
**Method**: Exhaustively search over predefined parameter grid

**Example**: SVM hyperparameters
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}
```

#### Random Search
**Method**: Randomly sample from parameter space

**Advantages**:
- More efficient than grid search
- Can find better solutions
- Less prone to local optima

#### Bayesian Optimization
**Method**: Use Bayesian models to guide search

**Process**:
1. **Build surrogate model** of objective function
2. **Use acquisition function** to select next point
3. **Update model** with new observations
4. **Repeat** until convergence

### Statistical Significance Testing
Testing whether observed differences are statistically significant.

#### Permutation Tests
**Method**: Randomly permute labels and compare statistics

**Example**: Testing if two groups have different means
1. **Compute observed difference** in means
2. **Permute group labels** randomly
3. **Compute difference** for permuted data
4. **Repeat** many times
5. **Compare observed** to permuted differences

#### Multiple Testing Correction
**Problem**: Multiple comparisons increase false positive rate

**Bonferroni Correction**: Divide α by number of tests
**False Discovery Rate (FDR)**: Control expected proportion of false positives

## Next Steps
This completes the comprehensive statistics tutorial series. You now have a solid foundation in:
- Descriptive statistics and data visualization
- Probability distributions and sampling
- Hypothesis testing and confidence intervals
- Correlation and regression analysis
- Analysis of variance
- Nonparametric statistics
- Advanced statistical methods

Continue practicing with real datasets and consider specialized courses in areas of interest such as machine learning, data science, or specific application domains.

## Additional Resources

### Software
- **R**: Comprehensive statistical computing
- **Python**: pandas, scipy, statsmodels, scikit-learn
- **SAS**: Advanced analytics
- **SPSS**: User-friendly interface
- **JMP**: Interactive statistical discovery

### Online Courses
- Coursera: Statistics and Data Science
- edX: Statistical Methods
- Khan Academy: Statistics
- MIT OpenCourseWare: Statistics

### Books
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Bayesian Data Analysis" by Gelman, Carlin, Stern, Rubin
- "Time Series Analysis and Its Applications" by Shumway, Stoffer
- "Multivariate Statistical Methods" by Johnson, Wichern

Remember: Statistics is a powerful tool for understanding data and making informed decisions. Continue learning and applying these methods to real-world problems!
