# Statistics Tutorial 02: Probability Distributions and Sampling

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand normal distribution and its properties
- Work with standard normal distribution and z-scores
- Understand sampling distributions
- Apply the central limit theorem
- Calculate probabilities using normal distribution
- Understand sampling methods and bias

## Normal Distribution

### Definition
The normal (Gaussian) distribution is a continuous probability distribution with bell-shaped curve.

**Probability Density Function (PDF)**:
f(x) = (1/σ√(2π)) e^(-(x-μ)²/(2σ²))

**Parameters**:
- μ = mean (center)
- σ = standard deviation (spread)

**Notation**: X ~ N(μ, σ²)

### Properties
1. **Symmetric**: Bell-shaped curve centered at μ
2. **68-95-99.7 Rule**: 
   - 68% within 1σ of μ
   - 95% within 2σ of μ
   - 99.7% within 3σ of μ
3. **Mean = Median = Mode**
4. **Asymptotic**: Never touches x-axis

## Standard Normal Distribution

### Definition
Normal distribution with μ = 0 and σ = 1.

**Notation**: Z ~ N(0, 1)

**PDF**: φ(z) = (1/√(2π)) e^(-z²/2)

### Z-Scores
Convert any normal distribution to standard normal:
z = (x - μ)/σ

**Example**: If X ~ N(100, 16), find P(X > 108)
1. Convert to z-score: z = (108 - 100)/4 = 2
2. P(X > 108) = P(Z > 2) = 1 - P(Z ≤ 2) = 1 - 0.9772 = 0.0228

### Using Standard Normal Table
- Find P(Z ≤ z) for given z
- Use symmetry: P(Z ≤ -z) = 1 - P(Z ≤ z)
- P(a ≤ Z ≤ b) = P(Z ≤ b) - P(Z ≤ a)

**Example**: Find P(-1.5 ≤ Z ≤ 2.0)
P(-1.5 ≤ Z ≤ 2.0) = P(Z ≤ 2.0) - P(Z ≤ -1.5)
= 0.9772 - (1 - 0.9332) = 0.9772 - 0.0668 = 0.9104

## Sampling Distributions

### Definition
The distribution of a statistic (like sample mean) computed from samples.

### Sample Mean Distribution
If X₁, X₂, ..., Xₙ are independent random variables from population with mean μ and variance σ², then:

**Sample Mean**: X̄ = (X₁ + X₂ + ... + Xₙ)/n

**Properties**:
- E[X̄] = μ
- Var(X̄) = σ²/n
- Standard Error: SE(X̄) = σ/√n

### Central Limit Theorem (CLT)
For large samples (n ≥ 30), the sampling distribution of X̄ is approximately normal:
X̄ ~ N(μ, σ²/n)

**Key Points**:
1. Works regardless of population distribution shape
2. Sample size n ≥ 30 is rule of thumb
3. Larger n gives better approximation

**Example**: Population mean = 50, σ = 10, n = 25
- E[X̄] = 50
- SE(X̄) = 10/√25 = 2
- X̄ ~ N(50, 4)
- P(X̄ > 52) = P(Z > (52-50)/2) = P(Z > 1) = 0.1587

## Sampling Methods

### Random Sampling
**Simple Random Sample**: Every subset of size n has equal probability of being selected.

**Stratified Sampling**: Divide population into strata, then sample from each stratum.

**Cluster Sampling**: Divide population into clusters, randomly select clusters, then sample all units in selected clusters.

**Systematic Sampling**: Select every kth unit from ordered list.

### Sampling Bias
**Selection Bias**: Some units more likely to be selected
**Nonresponse Bias**: Selected units don't respond
**Response Bias**: Respondents give inaccurate answers

### Sample Size Determination
For estimating population mean with margin of error E:
n = (z_(α/2) σ/E)²

**Example**: Want 95% confidence, σ = 10, E = 2
n = (1.96 × 10/2)² = (9.8)² ≈ 96

## Applications

### Quality Control
**Example**: Manufacturing process produces items with mean weight 100g, σ = 5g. Sample of 16 items taken.
- X̄ ~ N(100, 25/16) = N(100, 1.5625)
- P(X̄ < 98) = P(Z < (98-100)/1.25) = P(Z < -1.6) = 0.0548

### Survey Research
**Example**: Survey of 100 people, 60% support policy. What's probability that sample proportion > 65%?
- p̂ ~ N(0.6, 0.6×0.4/100) = N(0.6, 0.0024)
- P(p̂ > 0.65) = P(Z > (0.65-0.6)/√0.0024) = P(Z > 1.02) = 0.1539

## Practice Problems

### Problem 1
Test scores are normally distributed with μ = 75, σ = 10. Find the probability that a randomly selected score is between 70 and 85.

**Solution**:
P(70 < X < 85) = P((70-75)/10 < Z < (85-75)/10)
= P(-0.5 < Z < 1.0)
= P(Z < 1.0) - P(Z < -0.5)
= 0.8413 - 0.3085 = 0.5328

### Problem 2
A population has mean 50 and standard deviation 8. A sample of 36 is taken. Find P(X̄ > 52).

**Solution**:
X̄ ~ N(50, 64/36) = N(50, 1.78)
P(X̄ > 52) = P(Z > (52-50)/√1.78) = P(Z > 1.5) = 0.0668

### Problem 3
Heights of adult males are normally distributed with μ = 70 inches, σ = 3 inches. What height represents the 90th percentile?

**Solution**:
Find z such that P(Z ≤ z) = 0.90
From table: z = 1.28
x = μ + zσ = 70 + 1.28(3) = 73.84 inches

### Problem 4
A machine fills bottles with mean 500ml, σ = 10ml. Sample of 25 bottles taken. What's probability sample mean < 495ml?

**Solution**:
X̄ ~ N(500, 100/25) = N(500, 4)
P(X̄ < 495) = P(Z < (495-500)/2) = P(Z < -2.5) = 0.0062

## Key Takeaways
- Normal distribution is fundamental in statistics
- Z-scores standardize any normal distribution
- Sampling distributions describe statistic behavior
- Central limit theorem enables inference
- Proper sampling methods reduce bias
- Sample size affects precision of estimates

## Next Steps
In the next tutorial, we'll explore hypothesis testing, learning about null and alternative hypotheses, p-values, and different types of statistical tests.
