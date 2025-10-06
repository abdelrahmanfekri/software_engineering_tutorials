# Probability Tutorial 03: Continuous Probability Distributions

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand probability density functions
- Work with uniform and normal distributions
- Apply exponential and gamma distributions
- Use the beta distribution
- Calculate probabilities and expected values for continuous variables
- Apply continuous distributions to real-world problems

## Introduction to Continuous Distributions

### Probability Density Function (PDF)
For a continuous random variable X, the PDF is:
f(x) = dF(x)/dx

**Properties**:
1. f(x) ≥ 0 for all x
2. ∫ f(x) dx = 1 (integral over all possible values)

### Cumulative Distribution Function (CDF)
F(x) = P(X ≤ x) = ∫(-∞ to x) f(t) dt

### Relationship between PDF and CDF
- F'(x) = f(x)
- P(a < X < b) = F(b) - F(a) = ∫(a to b) f(x) dx

## Uniform Distribution

### Definition
All values in an interval are equally likely.

**PDF**: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise

**Parameters**:
- a = minimum value
- b = maximum value

**Properties**:
- E[X] = (a + b)/2
- Var(X) = (b - a)²/12

**Example**: Random number between 0 and 1
- f(x) = 1 for 0 ≤ x ≤ 1
- E[X] = 0.5
- Var(X) = 1/12 ≈ 0.083

### Applications
- Random sampling
- Monte Carlo simulations
- Computer graphics
- Quality control

## Normal (Gaussian) Distribution

### Definition
Bell-shaped curve symmetric about the mean.

**PDF**: f(x) = (1/σ√(2π)) e^(-(x-μ)²/(2σ²))

**Parameters**:
- μ = mean
- σ = standard deviation

**Properties**:
- E[X] = μ
- Var(X) = σ²
- Symmetric about μ
- 68-95-99.7 rule

### Standard Normal Distribution
When μ = 0 and σ = 1:
φ(x) = (1/√(2π)) e^(-x²/2)

**Standardization**: Z = (X - μ)/σ

### Applications
- Natural phenomena (heights, weights)
- Measurement errors
- Central limit theorem
- Statistical inference

**Example**: Heights of adult males (μ = 70", σ = 3")
- P(67" < X < 73") = P(-1 < Z < 1) ≈ 0.68
- P(X > 76") = P(Z > 2) ≈ 0.025

## Exponential Distribution

### Definition
Models waiting times between events in Poisson process.

**PDF**: f(x) = λe^(-λx) for x ≥ 0, 0 otherwise

**Parameters**:
- λ = rate parameter

**Properties**:
- E[X] = 1/λ
- Var(X) = 1/λ²
- Memoryless property

**Example**: Time between phone calls (λ = 2 per hour)
- E[X] = 0.5 hours
- P(X > 1) = e^(-2) ≈ 0.135

### Memoryless Property
P(X > s + t | X > s) = P(X > t)

### Applications
- Reliability engineering
- Queueing theory
- Radioactive decay
- Service times

## Gamma Distribution

### Definition
Generalization of exponential distribution.

**PDF**: f(x) = (λ^α/Γ(α)) x^(α-1) e^(-λx) for x > 0

Where Γ(α) = ∫(0 to ∞) t^(α-1) e^(-t) dt

**Parameters**:
- α = shape parameter
- λ = rate parameter

**Properties**:
- E[X] = α/λ
- Var(X) = α/λ²
- Special cases: α = 1 (exponential), α = n/2, λ = 1/2 (chi-square)

### Applications
- Insurance claims
- Rainfall modeling
- Reliability analysis
- Bayesian statistics

**Example**: Insurance claim amounts (α = 2, λ = 0.1)
- E[X] = 2/0.1 = 20
- Var(X) = 2/(0.1)² = 200

## Beta Distribution

### Definition
Models proportions and probabilities.

**PDF**: f(x) = (1/B(α,β)) x^(α-1) (1-x)^(β-1) for 0 ≤ x ≤ 1

Where B(α,β) = Γ(α)Γ(β)/Γ(α+β)

**Parameters**:
- α = shape parameter 1
- β = shape parameter 2

**Properties**:
- E[X] = α/(α + β)
- Var(X) = αβ/((α + β)²(α + β + 1))
- Range: [0, 1]

### Applications
- Bayesian inference
- Quality control
- Market share modeling
- A/B testing

**Example**: Success rate estimation (α = 5, β = 3)
- E[X] = 5/8 = 0.625
- Most likely value ≈ 0.6

## Distribution Relationships

### Central Limit Theorem
For large n, sample means approach normal distribution:
X̄ ≈ N(μ, σ²/n)

### Transformations
- If X ~ Uniform(0,1), then -ln(X) ~ Exponential(1)
- If X ~ Normal(0,1), then X² ~ Chi-square(1)
- If X ~ Gamma(α,λ), then 2λX ~ Chi-square(2α)

## Practice Problems

### Problem 1
A random variable X is uniformly distributed on [2, 8]. Find P(3 < X < 6).

**Solution**:
f(x) = 1/(8-2) = 1/6 for 2 ≤ x ≤ 8
P(3 < X < 6) = (6-3) × (1/6) = 3/6 = 1/2

### Problem 2
Test scores are normally distributed with mean 75 and standard deviation 10. What percentage scored above 85?

**Solution**:
P(X > 85) = P(Z > (85-75)/10) = P(Z > 1) ≈ 0.1587
About 15.87% scored above 85.

### Problem 3
The time between arrivals at a bank follows an exponential distribution with mean 5 minutes. What's the probability of waiting more than 10 minutes?

**Solution**:
λ = 1/5 = 0.2
P(X > 10) = e^(-0.2 × 10) = e^(-2) ≈ 0.135

### Problem 4
A manufacturing process produces items with lifetimes following a gamma distribution (α = 3, λ = 0.1). Find the expected lifetime and probability of lasting more than 20 units.

**Solution**:
E[X] = α/λ = 3/0.1 = 30
P(X > 20) = ∫(20 to ∞) (0.1³/Γ(3)) x² e^(-0.1x) dx
Using gamma CDF: P(X > 20) ≈ 0.323

## Key Takeaways
- Continuous distributions model uncountable outcomes
- PDFs integrate to 1 over their support
- Normal distribution is fundamental due to CLT
- Exponential distribution has memoryless property
- Gamma distribution generalizes exponential
- Beta distribution models proportions
- Each distribution has specific applications

## Next Steps
In the next tutorial, we'll explore joint probability distributions, learning about joint probability functions, marginal distributions, conditional distributions, and independence of random variables.
