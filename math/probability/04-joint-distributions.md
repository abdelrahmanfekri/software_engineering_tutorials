# Probability Tutorial 04: Joint Probability Distributions

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand joint probability mass and density functions
- Calculate marginal distributions
- Work with conditional distributions
- Determine independence of random variables
- Apply joint distributions to multivariate problems
- Use covariance and correlation measures

## Introduction to Joint Distributions

### Joint Probability Mass Function (Discrete)
For discrete random variables X and Y:
p(x,y) = P(X = x, Y = y)

**Properties**:
1. p(x,y) ≥ 0 for all (x,y)
2. Σ Σ p(x,y) = 1

### Joint Probability Density Function (Continuous)
For continuous random variables X and Y:
f(x,y) = ∂²F(x,y)/∂x∂y

**Properties**:
1. f(x,y) ≥ 0 for all (x,y)
2. ∫ ∫ f(x,y) dx dy = 1

### Joint Cumulative Distribution Function
F(x,y) = P(X ≤ x, Y ≤ y)

## Marginal Distributions

### Discrete Case
**Marginal PMF of X**: p_X(x) = Σ_y p(x,y)
**Marginal PMF of Y**: p_Y(y) = Σ_x p(x,y)

### Continuous Case
**Marginal PDF of X**: f_X(x) = ∫ f(x,y) dy
**Marginal PDF of Y**: f_Y(y) = ∫ f(x,y) dx

**Example**: Joint PMF table
```
     Y=0  Y=1  Y=2  p_X(x)
X=0  0.1  0.2  0.1   0.4
X=1  0.2  0.3  0.1   0.6
p_Y(y) 0.3  0.5  0.2   1.0
```

Marginal distributions:
- p_X(0) = 0.4, p_X(1) = 0.6
- p_Y(0) = 0.3, p_Y(1) = 0.5, p_Y(2) = 0.2

## Conditional Distributions

### Discrete Case
**Conditional PMF**: p_X|Y(x|y) = p(x,y)/p_Y(y) for p_Y(y) > 0

### Continuous Case
**Conditional PDF**: f_X|Y(x|y) = f(x,y)/f_Y(y) for f_Y(y) > 0

**Example**: From the joint PMF above
- p_X|Y(0|1) = p(0,1)/p_Y(1) = 0.2/0.5 = 0.4
- p_X|Y(1|1) = p(1,1)/p_Y(1) = 0.3/0.5 = 0.6

## Independence of Random Variables

### Definition
Random variables X and Y are independent if:
f(x,y) = f_X(x) f_Y(y) for all (x,y)

### Equivalent Conditions
1. F(x,y) = F_X(x) F_Y(y)
2. f_X|Y(x|y) = f_X(x) for all y
3. f_Y|X(y|x) = f_Y(y) for all x

**Example**: Are X and Y independent in the previous example?
- p(0,0) = 0.1, p_X(0) × p_Y(0) = 0.4 × 0.3 = 0.12 ≠ 0.1
- Therefore, X and Y are NOT independent

## Multivariate Normal Distribution

### Bivariate Normal
**PDF**: f(x,y) = (1/(2πσ_Xσ_Y√(1-ρ²))) × exp(-Q/2)

Where Q = (1/(1-ρ²))[(x-μ_X)²/σ_X² - 2ρ(x-μ_X)(y-μ_Y)/(σ_Xσ_Y) + (y-μ_Y)²/σ_Y²]

**Parameters**:
- μ_X, μ_Y = means
- σ_X, σ_Y = standard deviations
- ρ = correlation coefficient

**Properties**:
- Marginal distributions are normal
- Conditional distributions are normal
- Independence ⇔ ρ = 0

## Covariance and Correlation

### Covariance
Cov(X,Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]

### Correlation Coefficient
ρ = Corr(X,Y) = Cov(X,Y)/(σ_X σ_Y)

**Properties**:
- -1 ≤ ρ ≤ 1
- ρ = 1: perfect positive linear relationship
- ρ = -1: perfect negative linear relationship
- ρ = 0: no linear relationship (but may have nonlinear relationship)

### Properties of Covariance
1. Cov(X,X) = Var(X)
2. Cov(X,Y) = Cov(Y,X)
3. Cov(aX + b, cY + d) = ac Cov(X,Y)
4. Cov(X + Y, Z) = Cov(X,Z) + Cov(Y,Z)

## Multinomial Distribution

### Definition
Generalization of binomial distribution for multiple categories.

**PMF**: P(X₁ = x₁, ..., X_k = x_k) = (n!/(x₁!...x_k!)) p₁^x₁ ... p_k^x_k

**Parameters**:
- n = number of trials
- p₁, ..., p_k = probabilities of categories

**Properties**:
- E[X_i] = np_i
- Var(X_i) = np_i(1-p_i)
- Cov(X_i, X_j) = -np_i p_j

**Example**: Rolling die 10 times, counting each face
- n = 10, p₁ = ... = p₆ = 1/6
- P(X₁ = 2, X₂ = 1, ..., X₆ = 1) = (10!/(2!1!...1!)) (1/6)¹⁰

## Applications

### Quality Control
- Joint distribution of multiple quality measures
- Independence testing between processes

### Finance
- Joint returns of multiple assets
- Portfolio risk analysis
- Correlation analysis

### Medicine
- Joint effects of treatments
- Risk factor interactions
- Clinical trial analysis

### Engineering
- Multiple stress factors
- Reliability analysis
- System performance

## Practice Problems

### Problem 1
Given joint PMF:
```
     Y=1  Y=2  Y=3
X=1  0.1  0.2  0.1
X=2  0.2  0.3  0.1
```
Find marginal distributions and check independence.

**Solution**:
Marginal distributions:
- p_X(1) = 0.4, p_X(2) = 0.6
- p_Y(1) = 0.3, p_Y(2) = 0.5, p_Y(3) = 0.2

Independence check:
- p(1,1) = 0.1, p_X(1) × p_Y(1) = 0.4 × 0.3 = 0.12 ≠ 0.1
- X and Y are NOT independent

### Problem 2
For continuous random variables with joint PDF:
f(x,y) = 6xy for 0 ≤ x ≤ 1, 0 ≤ y ≤ 1, x + y ≤ 1
Find marginal PDFs and conditional PDF f_X|Y(x|y).

**Solution**:
Marginal PDF of Y:
f_Y(y) = ∫(0 to 1-y) 6xy dx = 6y[x²/2](0 to 1-y) = 3y(1-y)²

Conditional PDF:
f_X|Y(x|y) = f(x,y)/f_Y(y) = 6xy/(3y(1-y)²) = 2x/(1-y)²

### Problem 3
Calculate covariance for the discrete example in Problem 1.

**Solution**:
E[X] = 1(0.4) + 2(0.6) = 1.6
E[Y] = 1(0.3) + 2(0.5) + 3(0.2) = 1.9
E[XY] = 1(1)(0.1) + 1(2)(0.2) + 1(3)(0.1) + 2(1)(0.2) + 2(2)(0.3) + 2(3)(0.1)
= 0.1 + 0.4 + 0.3 + 0.4 + 1.2 + 0.6 = 3.0

Cov(X,Y) = E[XY] - E[X]E[Y] = 3.0 - 1.6(1.9) = 3.0 - 3.04 = -0.04

### Problem 4
In a bivariate normal distribution with μ_X = 0, μ_Y = 0, σ_X = 1, σ_Y = 2, ρ = 0.5, find P(X > 1, Y > 2).

**Solution**:
Standardize: Z_X = X, Z_Y = (Y - 0)/2 = Y/2
P(X > 1, Y > 2) = P(Z_X > 1, Z_Y > 1)
Using bivariate normal tables or simulation: ≈ 0.067

## Key Takeaways
- Joint distributions describe multiple random variables
- Marginal distributions are obtained by summing/integrating
- Conditional distributions update beliefs given information
- Independence simplifies joint distributions
- Covariance measures linear relationship
- Correlation is normalized covariance
- Multivariate normal is fundamental for applications

## Next Steps
In the next tutorial, we'll explore expected values and moments, learning about mean, variance, higher moments, moment generating functions, and characteristic functions.
