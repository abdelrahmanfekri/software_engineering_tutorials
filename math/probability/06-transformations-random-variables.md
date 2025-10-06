# Probability Tutorial 06: Transformations of Random Variables

## Learning Objectives
By the end of this tutorial, you will be able to:
- Transform random variables using different methods
- Apply the method of distribution functions
- Use the method of transformations
- Apply the method of moment generating functions
- Work with order statistics
- Handle multivariate transformations

## Introduction to Transformations

### Why Transform Random Variables?
- Simplify calculations
- Standardize distributions
- Create new distributions
- Analyze relationships between variables

### Common Transformations
- Linear: Y = aX + b
- Power: Y = X^n
- Exponential: Y = e^X
- Logarithmic: Y = ln(X)
- Trigonometric: Y = sin(X), cos(X)

## Method of Distribution Functions

### For Continuous Random Variables
1. Find the CDF of Y: F_Y(y) = P(Y ≤ y)
2. Express in terms of X: F_Y(y) = P(g(X) ≤ y)
3. Solve for X: F_Y(y) = P(X ≤ g^(-1)(y))
4. Differentiate to get PDF: f_Y(y) = dF_Y(y)/dy

**Example**: Y = X² where X ~ Uniform(-1, 1)
- F_Y(y) = P(X² ≤ y) = P(-√y ≤ X ≤ √y) = √y for 0 ≤ y ≤ 1
- f_Y(y) = 1/(2√y) for 0 ≤ y ≤ 1

### For Discrete Random Variables
1. List all possible values of Y
2. Find probabilities: P(Y = y) = P(X ∈ {x: g(x) = y})
3. Sum probabilities for each y

**Example**: Y = X² where X ~ {-1, 0, 1} with equal probability
- Y takes values {0, 1}
- P(Y = 0) = P(X = 0) = 1/3
- P(Y = 1) = P(X = -1) + P(X = 1) = 1/3 + 1/3 = 2/3

## Method of Transformations (Change of Variables)

### One-to-One Transformations
For Y = g(X) where g is strictly monotonic:

f_Y(y) = f_X(g^(-1)(y)) |d/dy g^(-1)(y)|

### Multivariate Case
For Y₁ = g₁(X₁, X₂), Y₂ = g₂(X₁, X₂):

f_Y₁,Y₂(y₁, y₂) = f_X₁,X₂(x₁, x₂) |J|

Where J is the Jacobian determinant:
J = det(∂x₁/∂y₁  ∂x₁/∂y₂)
    (∂x₂/∂y₁  ∂x₂/∂y₂)

**Example**: Box-Muller transformation
X₁, X₂ ~ Uniform(0,1) independent
Y₁ = √(-2ln(X₁)) cos(2πX₂)
Y₂ = √(-2ln(X₁)) sin(2πX₂)

Then Y₁, Y₂ ~ Normal(0,1) independent

## Method of Moment Generating Functions

### Using MGFs to Find Distributions
1. Find MGF of transformed variable
2. Compare with known MGFs
3. Identify the distribution

**Example**: Y = aX + b
M_Y(t) = E[e^(tY)] = E[e^(t(aX+b))] = e^(bt) E[e^(atX)] = e^(bt) M_X(at)

If X ~ Normal(μ, σ²), then Y ~ Normal(aμ + b, a²σ²)

### Sums of Independent Random Variables
If X₁, X₂, ..., Xₙ are independent:
M_(X₁+X₂+...+Xₙ)(t) = M_X₁(t) M_X₂(t) ... M_Xₙ(t)

**Example**: Sum of independent Poisson variables
If X₁ ~ Poisson(λ₁) and X₂ ~ Poisson(λ₂), then X₁ + X₂ ~ Poisson(λ₁ + λ₂)

## Order Statistics

### Definition
For random variables X₁, X₂, ..., Xₙ, the order statistics are:
X_(1) ≤ X_(2) ≤ ... ≤ X_(n)

Where X_(k) is the kth smallest value.

### Distribution of Order Statistics

**Minimum**: X_(1)
F_(1)(x) = 1 - (1 - F(x))^n
f_(1)(x) = n(1 - F(x))^(n-1) f(x)

**Maximum**: X_(n)
F_(n)(x) = F(x)^n
f_(n)(x) = nF(x)^(n-1) f(x)

**kth Order Statistic**: X_(k)
f_(k)(x) = n!/((k-1)!(n-k)!) F(x)^(k-1) (1-F(x))^(n-k) f(x)

**Example**: Uniform(0,1) sample of size n
- X_(1) ~ Beta(1, n)
- X_(n) ~ Beta(n, 1)
- X_(k) ~ Beta(k, n-k+1)

## Special Transformations

### Standardization
Z = (X - μ)/σ transforms any distribution to have mean 0 and variance 1.

### Probability Integral Transform
If X has CDF F, then F(X) ~ Uniform(0,1).

**Inverse**: If U ~ Uniform(0,1), then F^(-1)(U) has CDF F.

### Log-Normal Distribution
If X ~ Normal(μ, σ²), then Y = e^X ~ LogNormal(μ, σ²)

**PDF**: f_Y(y) = (1/(yσ√(2π))) e^(-(ln(y)-μ)²/(2σ²)) for y > 0

### Chi-Square Distribution
If Z₁, Z₂, ..., Zₙ ~ Normal(0,1) independent, then:
χ² = Z₁² + Z₂² + ... + Zₙ² ~ Chi-square(n)

**Properties**:
- E[χ²] = n
- Var(χ²) = 2n
- Chi-square(k) = Gamma(k/2, 1/2)

## Multivariate Transformations

### Linear Transformations
For Y = AX + b where A is a matrix:
- E[Y] = AE[X] + b
- Cov(Y) = A Cov(X) A^T

### Polar Coordinates
For (X, Y) ~ Normal(0, I₂):
R = √(X² + Y²), Θ = arctan(Y/X)
- R² ~ Chi-square(2)
- Θ ~ Uniform(0, 2π)
- R and Θ are independent

## Applications

### Statistical Inference
- Standardization for hypothesis testing
- Confidence interval construction
- Sample size calculations

### Simulation
- Generating random variables from arbitrary distributions
- Monte Carlo methods
- Bootstrap techniques

### Engineering
- Signal processing
- Control systems
- Reliability analysis

### Finance
- Risk modeling
- Option pricing
- Portfolio optimization

## Practice Problems

### Problem 1
If X ~ Uniform(0, 1), find the distribution of Y = -ln(X).

**Solution**:
F_Y(y) = P(-ln(X) ≤ y) = P(ln(X) ≥ -y) = P(X ≥ e^(-y)) = 1 - e^(-y)
f_Y(y) = e^(-y) for y > 0
Therefore, Y ~ Exponential(1)

### Problem 2
If X ~ Normal(0, 1), find the distribution of Y = X².

**Solution**:
For y > 0: F_Y(y) = P(X² ≤ y) = P(-√y ≤ X ≤ √y) = 2Φ(√y) - 1
f_Y(y) = d/dy[2Φ(√y) - 1] = 2φ(√y)(1/(2√y)) = φ(√y)/√y
= (1/√(2π)) e^(-y/2) / √y = (1/√(2πy)) e^(-y/2)
Therefore, Y ~ Chi-square(1)

### Problem 3
Find the distribution of the minimum of n independent Exponential(λ) random variables.

**Solution**:
For X₁, ..., Xₙ ~ Exponential(λ) independent:
F_(1)(x) = 1 - (1 - F(x))^n = 1 - (e^(-λx))^n = 1 - e^(-nλx)
f_(1)(x) = nλe^(-nλx) for x > 0
Therefore, X_(1) ~ Exponential(nλ)

### Problem 4
If X₁, X₂ ~ Normal(0, 1) independent, find the distribution of Y = X₁/X₂.

**Solution**:
This is a Cauchy distribution. Using the method of transformations:
Let Y₁ = X₁/X₂, Y₂ = X₂
Then X₁ = Y₁Y₂, X₂ = Y₂
Jacobian: J = |Y₂|
f_Y₁,Y₂(y₁, y₂) = f_X₁,X₂(y₁y₂, y₂) |y₂|
= (1/(2π)) e^(-(y₁²y₂² + y₂²)/2) |y₂|
Integrating over y₂: f_Y₁(y₁) = 1/(π(1 + y₁²))
Therefore, Y ~ Cauchy(0, 1)

## Key Takeaways
- Transformations create new distributions from existing ones
- Method of distribution functions works for any transformation
- Method of transformations requires monotonic functions
- MGFs are useful for sums of independent variables
- Order statistics have specific distributions
- Multivariate transformations require Jacobians
- Common transformations have well-known results

## Next Steps
In the next tutorial, we'll explore limit theorems, learning about the law of large numbers, central limit theorem, and various types of convergence in probability theory.
