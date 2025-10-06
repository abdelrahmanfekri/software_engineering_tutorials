# Probability Tutorial 05: Expected Values and Moments

## Learning Objectives
By the end of this tutorial, you will be able to:
- Calculate expected values for discrete and continuous distributions
- Understand variance and standard deviation
- Work with higher moments (skewness, kurtosis)
- Use moment generating functions
- Apply characteristic functions
- Calculate expected values of functions of random variables

## Expected Value (Mean)

### Discrete Case
E[X] = Σ x p(x)

### Continuous Case
E[X] = ∫ x f(x) dx

### Properties of Expected Value
1. E[aX + b] = aE[X] + b (linearity)
2. E[X + Y] = E[X] + E[Y] (additivity)
3. E[XY] = E[X]E[Y] if X and Y are independent
4. E[g(X)] = Σ g(x) p(x) or ∫ g(x) f(x) dx

**Example**: Rolling a fair die
E[X] = 1(1/6) + 2(1/6) + ... + 6(1/6) = 21/6 = 3.5

**Example**: Exponential distribution with λ = 2
E[X] = ∫(0 to ∞) x (2e^(-2x)) dx = 1/2

## Variance and Standard Deviation

### Definition
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²

### Standard Deviation
σ = √Var(X)

### Properties of Variance
1. Var(aX + b) = a²Var(X)
2. Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
3. Var(X + Y) = Var(X) + Var(Y) if X and Y are independent
4. Var(X) ≥ 0

**Example**: Binomial distribution (n, p)
- E[X] = np
- Var(X) = np(1-p)

**Example**: Normal distribution (μ, σ²)
- E[X] = μ
- Var(X) = σ²

## Higher Moments

### Raw Moments
μ'_k = E[X^k]

### Central Moments
μ_k = E[(X - E[X])^k]

### Skewness
γ₁ = μ₃/σ³ = E[(X - μ)³]/σ³

**Interpretation**:
- γ₁ > 0: right-skewed (tail extends right)
- γ₁ < 0: left-skewed (tail extends left)
- γ₁ = 0: symmetric

### Kurtosis
γ₂ = μ₄/σ⁴ - 3 = E[(X - μ)⁴]/σ⁴ - 3

**Interpretation**:
- γ₂ > 0: heavy tails (leptokurtic)
- γ₂ < 0: light tails (platykurtic)
- γ₂ = 0: normal kurtosis (mesokurtic)

**Example**: Normal distribution
- Skewness = 0 (symmetric)
- Kurtosis = 0 (mesokurtic)

## Moment Generating Functions

### Definition
M_X(t) = E[e^(tX)]

### Properties
1. M_X(0) = 1
2. M'_X(0) = E[X]
3. M''_X(0) = E[X²]
4. M^(k)_X(0) = E[X^k]
5. M_(aX+b)(t) = e^(bt) M_X(at)

### Uniqueness Theorem
If MGFs exist and are equal, then the distributions are equal.

**Example**: Exponential distribution with λ
M_X(t) = E[e^(tX)] = ∫(0 to ∞) e^(tx) λe^(-λx) dx
= λ/(λ-t) for t < λ

M'_X(t) = λ/(λ-t)²
E[X] = M'_X(0) = λ/λ² = 1/λ

## Characteristic Functions

### Definition
φ_X(t) = E[e^(itX)] where i = √(-1)

### Properties
1. φ_X(0) = 1
2. |φ_X(t)| ≤ 1
3. φ_X(-t) = φ̄_X(t) (complex conjugate)
4. φ_(aX+b)(t) = e^(ibt) φ_X(at)

### Advantages over MGF
- Always exists (even when MGF doesn't)
- Used in central limit theorem proofs
- Fourier transform relationship

**Example**: Standard normal distribution
φ_X(t) = e^(-t²/2)

## Functions of Random Variables

### Expected Value of g(X)
E[g(X)] = Σ g(x) p(x) or ∫ g(x) f(x) dx

### Variance of g(X)
Var(g(X)) = E[g(X)²] - (E[g(X)])²

**Example**: If X ~ Uniform(0,1), find E[X²] and Var(X²)
E[X²] = ∫(0 to 1) x² dx = [x³/3](0 to 1) = 1/3
E[X⁴] = ∫(0 to 1) x⁴ dx = [x⁵/5](0 to 1) = 1/5
Var(X²) = E[X⁴] - (E[X²])² = 1/5 - (1/3)² = 1/5 - 1/9 = 4/45

## Conditional Expectation

### Definition
E[X|Y = y] = Σ x p_X|Y(x|y) or ∫ x f_X|Y(x|y) dx

### Properties
1. E[E[X|Y]] = E[X] (law of total expectation)
2. E[X|Y] is a function of Y
3. E[g(X)|Y] = ∫ g(x) f_X|Y(x|y) dx

**Example**: Joint PMF
```
     Y=1  Y=2
X=1  0.2  0.3
X=2  0.3  0.2
```

E[X|Y = 1] = 1(0.2/0.5) + 2(0.3/0.5) = 0.4 + 1.2 = 1.6
E[X|Y = 2] = 1(0.3/0.5) + 2(0.2/0.5) = 0.6 + 0.8 = 1.4

## Applications

### Risk Assessment
- Expected loss calculations
- Value at Risk (VaR)
- Portfolio optimization

### Quality Control
- Process capability indices
- Control chart limits
- Acceptance sampling

### Engineering
- Reliability analysis
- Signal processing
- System performance

### Finance
- Option pricing
- Risk management
- Portfolio theory

## Practice Problems

### Problem 1
A random variable X has PMF: p(1) = 0.3, p(2) = 0.5, p(3) = 0.2
Find E[X], Var(X), and E[X²].

**Solution**:
E[X] = 1(0.3) + 2(0.5) + 3(0.2) = 0.3 + 1.0 + 0.6 = 1.9
E[X²] = 1²(0.3) + 2²(0.5) + 3²(0.2) = 0.3 + 2.0 + 1.8 = 4.1
Var(X) = E[X²] - (E[X])² = 4.1 - (1.9)² = 4.1 - 3.61 = 0.49

### Problem 2
For X ~ Normal(μ = 10, σ² = 4), find E[X²] and Var(X²).

**Solution**:
E[X²] = Var(X) + (E[X])² = 4 + 10² = 104
For X² ~ Chi-square(1) scaled: Var(X²) = 2σ⁴ = 2(16) = 32

### Problem 3
Find the MGF of Poisson(λ) and use it to find E[X] and Var(X).

**Solution**:
M_X(t) = E[e^(tX)] = Σ(k=0 to ∞) e^(tk) (λ^k e^(-λ))/k!
= e^(-λ) Σ(k=0 to ∞) (λe^t)^k/k! = e^(-λ) e^(λe^t) = e^(λ(e^t - 1))

M'_X(t) = λe^t e^(λ(e^t - 1))
E[X] = M'_X(0) = λe^0 e^(λ(e^0 - 1)) = λ

M''_X(t) = λe^t e^(λ(e^t - 1)) + λ²e^(2t) e^(λ(e^t - 1))
E[X²] = M''_X(0) = λ + λ²
Var(X) = E[X²] - (E[X])² = λ + λ² - λ² = λ

### Problem 4
If X and Y are independent with E[X] = 2, Var(X) = 3, E[Y] = 1, Var(Y) = 4, find E[2X - 3Y + 5] and Var(2X - 3Y + 5).

**Solution**:
E[2X - 3Y + 5] = 2E[X] - 3E[Y] + 5 = 2(2) - 3(1) + 5 = 6
Var(2X - 3Y + 5) = 2²Var(X) + (-3)²Var(Y) = 4(3) + 9(4) = 48

## Key Takeaways
- Expected value is the center of a distribution
- Variance measures spread around the mean
- Higher moments describe shape characteristics
- MGFs uniquely determine distributions
- Characteristic functions always exist
- Conditional expectation follows law of total expectation
- Linearity properties simplify calculations

## Next Steps
In the next tutorial, we'll explore transformations of random variables, learning about functions of random variables, methods for finding distributions, and order statistics.
