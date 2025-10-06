# Probability Tutorial 07: Limit Theorems

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the law of large numbers
- Apply the central limit theorem
- Work with convergence in probability
- Understand convergence in distribution
- Apply limit theorems to statistical inference
- Use approximations for practical problems

## Introduction to Limit Theorems

### Why Limit Theorems Matter
- Provide approximations for large samples
- Justify statistical methods
- Connect probability theory to statistics
- Enable practical applications

### Types of Convergence
1. **Almost Sure Convergence**: Xₙ → X a.s.
2. **Convergence in Probability**: Xₙ → X in probability
3. **Convergence in Distribution**: Xₙ → X in distribution
4. **Convergence in Lᵖ**: E[|Xₙ - X|ᵖ] → 0

## Weak Law of Large Numbers

### Statement
Let X₁, X₂, ..., Xₙ be independent and identically distributed (i.i.d.) random variables with E[Xᵢ] = μ < ∞. Then:

X̄ₙ = (X₁ + X₂ + ... + Xₙ)/n → μ in probability

### Interpretation
As sample size increases, the sample mean converges to the population mean.

**Example**: Rolling a die
- True mean: μ = 3.5
- Sample means approach 3.5 as n increases
- For large n, P(|X̄ₙ - 3.5| > ε) → 0

### Proof Sketch
Using Chebyshev's inequality:
P(|X̄ₙ - μ| ≥ ε) ≤ Var(X̄ₙ)/ε² = σ²/(nε²) → 0 as n → ∞

## Strong Law of Large Numbers

### Statement
Under the same conditions as WLLN, but with E[|Xᵢ|] < ∞:

X̄ₙ → μ almost surely

### Difference from WLLN
- WLLN: convergence in probability
- SLLN: convergence almost surely (stronger)
- SLLN implies WLLN

## Central Limit Theorem

### Statement
Let X₁, X₂, ..., Xₙ be i.i.d. with E[Xᵢ] = μ and Var(Xᵢ) = σ² < ∞. Then:

√n(X̄ₙ - μ)/σ → N(0, 1) in distribution

### Alternative Forms
1. X̄ₙ ≈ N(μ, σ²/n) for large n
2. (X₁ + ... + Xₙ) ≈ N(nμ, nσ²) for large n
3. Standardized: (X̄ₙ - μ)/(σ/√n) ≈ N(0, 1)

### Applications
- Confidence intervals
- Hypothesis testing
- Sample size calculations
- Quality control

**Example**: Sample mean of 100 observations
- If population mean = 50, variance = 25
- X̄₁₀₀ ≈ N(50, 25/100) = N(50, 0.25)
- P(X̄₁₀₀ > 51) ≈ P(Z > (51-50)/0.5) = P(Z > 2) ≈ 0.025

## Convergence in Probability

### Definition
Xₙ → X in probability if for all ε > 0:
lim(n→∞) P(|Xₙ - X| ≥ ε) = 0

### Properties
1. If Xₙ → X and Yₙ → Y in probability, then:
   - Xₙ + Yₙ → X + Y
   - XₙYₙ → XY
   - g(Xₙ) → g(X) if g is continuous

2. Convergence in probability implies convergence in distribution

**Example**: Xₙ = X + εₙ where εₙ → 0 in probability
Then Xₙ → X in probability

## Convergence in Distribution

### Definition
Xₙ → X in distribution if:
lim(n→∞) Fₙ(x) = F(x) at all continuity points of F

### Properties
1. If Xₙ → X in distribution and Yₙ → c (constant), then:
   - Xₙ + Yₙ → X + c
   - XₙYₙ → cX
   - Xₙ/Yₙ → X/c if c ≠ 0

2. Continuous mapping theorem: If Xₙ → X and g is continuous, then g(Xₙ) → g(X)

### Slutsky's Theorem
If Xₙ → X in distribution and Yₙ → c in probability, then:
- Xₙ + Yₙ → X + c
- XₙYₙ → cX
- Xₙ/Yₙ → X/c if c ≠ 0

## Delta Method

### Statement
If √n(Xₙ - θ) → N(0, σ²) and g is differentiable at θ, then:
√n(g(Xₙ) - g(θ)) → N(0, (g'(θ))²σ²)

### Applications
- Variance of transformed estimators
- Confidence intervals for functions of parameters
- Asymptotic distributions

**Example**: If X̄ₙ ≈ N(μ, σ²/n), then:
√n(ln(X̄ₙ) - ln(μ)) → N(0, σ²/μ²)

## Berry-Esseen Theorem

### Statement
Provides rate of convergence in CLT:
|P(√n(X̄ₙ - μ)/σ ≤ x) - Φ(x)| ≤ C E[|X₁ - μ|³]/(σ³√n)

Where C is a universal constant (C ≤ 0.4748).

### Interpretation
- Error in normal approximation is O(1/√n)
- Depends on third moment (skewness)
- Provides bounds on approximation quality

## Applications in Statistics

### Confidence Intervals
For large samples, 95% CI for μ:
X̄ₙ ± 1.96 σ/√n

### Hypothesis Testing
Test H₀: μ = μ₀ vs H₁: μ ≠ μ₀
Test statistic: Z = √n(X̄ₙ - μ₀)/σ
Reject H₀ if |Z| > 1.96

### Sample Size Calculations
To achieve margin of error m with confidence level 1-α:
n = (z_(α/2) σ/m)²

## Multivariate Central Limit Theorem

### Statement
Let X₁, X₂, ..., Xₙ be i.i.d. random vectors with E[Xᵢ] = μ and Cov(Xᵢ) = Σ. Then:

√n(X̄ₙ - μ) → N(0, Σ) in distribution

### Applications
- Multivariate hypothesis testing
- Principal component analysis
- Multivariate confidence regions

## Practice Problems

### Problem 1
A fair coin is flipped 1000 times. Use CLT to approximate the probability of getting between 480 and 520 heads.

**Solution**:
X ~ Binomial(1000, 0.5)
E[X] = 500, Var(X) = 250
Using CLT: X ≈ N(500, 250)
P(480 ≤ X ≤ 520) = P((480-500)/√250 ≤ Z ≤ (520-500)/√250)
= P(-1.26 ≤ Z ≤ 1.26) ≈ 0.79

### Problem 2
The lifetime of light bulbs has mean 1000 hours and standard deviation 200 hours. What's the probability that the average lifetime of 25 bulbs exceeds 1050 hours?

**Solution**:
Using CLT: X̄₂₅ ≈ N(1000, 200²/25) = N(1000, 1600)
P(X̄₂₅ > 1050) = P(Z > (1050-1000)/40) = P(Z > 1.25) ≈ 0.106

### Problem 3
Use the delta method to find the asymptotic distribution of √n(1/X̄ₙ - 1/μ) when X̄ₙ ≈ N(μ, σ²/n).

**Solution**:
Let g(x) = 1/x, so g'(x) = -1/x²
By delta method: √n(1/X̄ₙ - 1/μ) → N(0, σ²/μ⁴)

### Problem 4
Show that if Xₙ → X in probability and Yₙ → Y in probability, then Xₙ + Yₙ → X + Y in probability.

**Solution**:
For any ε > 0:
P(|Xₙ + Yₙ - (X + Y)| ≥ ε) ≤ P(|Xₙ - X| ≥ ε/2) + P(|Yₙ - Y| ≥ ε/2)
Both terms → 0 as n → ∞, so the sum → 0.

## Key Takeaways
- Law of large numbers: sample means converge to population mean
- Central limit theorem: sample means are approximately normal
- Convergence concepts are fundamental to asymptotic theory
- Delta method extends CLT to functions of estimators
- Berry-Esseen provides convergence rates
- Limit theorems justify many statistical methods
- Multivariate extensions handle vector-valued data

## Next Steps
In the next tutorial, we'll explore stochastic processes, learning about Markov chains, Poisson processes, Brownian motion, and their applications in various fields.
