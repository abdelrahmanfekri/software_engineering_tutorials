# Statistics Tutorial 03: Hypothesis Testing

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand null and alternative hypotheses
- Calculate and interpret p-values
- Distinguish between Type I and Type II errors
- Perform one-sample and two-sample tests
- Conduct chi-square tests
- Choose appropriate statistical tests
- Interpret test results correctly

## Introduction to Hypothesis Testing

### What is Hypothesis Testing?
Hypothesis testing is a statistical method used to make decisions about population parameters based on sample data. It provides a framework for testing claims about populations using evidence from samples.

### The Hypothesis Testing Process
1. **State the hypotheses** (null and alternative)
2. **Choose significance level** (α)
3. **Calculate test statistic**
4. **Find p-value**
5. **Make decision** (reject or fail to reject null)
6. **Draw conclusion**

## Null and Alternative Hypotheses

### Null Hypothesis (H₀)
- Statement of "no effect" or "no difference"
- Assumed to be true unless evidence suggests otherwise
- Usually contains equality (=, ≤, ≥)

### Alternative Hypothesis (H₁ or Hₐ)
- Statement of "effect" or "difference"
- What we want to prove
- Usually contains inequality (≠, <, >)

### Types of Tests
1. **Two-tailed test**: H₀: μ = μ₀ vs H₁: μ ≠ μ₀
2. **One-tailed test (left)**: H₀: μ ≥ μ₀ vs H₁: μ < μ₀
3. **One-tailed test (right)**: H₀: μ ≤ μ₀ vs H₁: μ > μ₀

**Example**: Testing if new teaching method improves test scores
- H₀: μ = 75 (no improvement)
- H₁: μ > 75 (improvement)

## Type I and Type II Errors

### Type I Error (α)
- Rejecting H₀ when it's true
- Probability = α (significance level)
- "False positive"

### Type II Error (β)
- Failing to reject H₀ when it's false
- Probability = β
- "False negative"

### Power of Test
- Power = 1 - β
- Probability of correctly rejecting false H₀
- Higher power = better test

### Error Trade-offs
- Decreasing α increases β
- Increasing sample size decreases both α and β
- Choose α based on consequences of Type I error

## Significance Levels and P-values

### Significance Level (α)
- Maximum probability of Type I error
- Common values: 0.05, 0.01, 0.10
- α = 0.05 means 5% chance of false positive

### P-value
- Probability of observing test statistic as extreme or more extreme than observed, assuming H₀ is true
- Smaller p-value = stronger evidence against H₀
- p < α → reject H₀
- p ≥ α → fail to reject H₀

### Interpreting P-values
- p < 0.01: Very strong evidence against H₀
- 0.01 ≤ p < 0.05: Strong evidence against H₀
- 0.05 ≤ p < 0.10: Weak evidence against H₀
- p ≥ 0.10: Little or no evidence against H₀

## One-Sample Tests

### One-Sample t-test
Tests whether population mean equals specified value.

**Assumptions**:
- Data is approximately normal
- Random sampling
- Independent observations

**Test Statistic**:
t = (x̄ - μ₀)/(s/√n)

**Example**: Test if average height is 70 inches
- Sample: n = 25, x̄ = 71.2, s = 2.5
- H₀: μ = 70, H₁: μ ≠ 70
- t = (71.2 - 70)/(2.5/√25) = 1.2/0.5 = 2.4
- df = 24, p-value ≈ 0.025
- Since p < 0.05, reject H₀

### One-Sample z-test
When population standard deviation is known.

**Test Statistic**:
z = (x̄ - μ₀)/(σ/√n)

**Example**: Test if average IQ is 100
- Population σ = 15, n = 36, x̄ = 105
- H₀: μ = 100, H₁: μ > 100
- z = (105 - 100)/(15/√36) = 5/2.5 = 2.0
- p-value = P(Z > 2.0) = 0.0228
- Since p < 0.05, reject H₀

## Two-Sample Tests

### Two-Sample t-test (Independent)
Compares means of two independent groups.

**Assumptions**:
- Both samples are approximately normal
- Independent random sampling
- Equal or unequal variances

**Test Statistic**:
t = (x̄₁ - x̄₂)/SE(x̄₁ - x̄₂)

**Example**: Compare test scores of two teaching methods
- Method A: n₁ = 30, x̄₁ = 78, s₁ = 8
- Method B: n₂ = 25, x̄₂ = 82, s₂ = 7
- H₀: μ₁ = μ₂, H₁: μ₁ ≠ μ₂
- SE = √(s₁²/n₁ + s₂²/n₂) = √(64/30 + 49/25) = √(2.13 + 1.96) = 2.02
- t = (78 - 82)/2.02 = -1.98
- df ≈ 53, p-value ≈ 0.053
- Since p > 0.05, fail to reject H₀

### Paired t-test
Compares means of paired observations.

**Test Statistic**:
t = d̄/(s_d/√n)

Where d̄ is mean difference and s_d is standard deviation of differences.

**Example**: Test effectiveness of diet program
- Before: 180, 175, 190, 185, 170
- After: 175, 170, 185, 180, 165
- Differences: -5, -5, -5, -5, -5
- d̄ = -5, s_d = 0
- H₀: μ_d = 0, H₁: μ_d < 0
- t = -5/(0/√5) = undefined (perfect correlation)
- In practice, would use actual data with variation

## Chi-Square Tests

### Chi-Square Goodness-of-Fit Test
Tests whether observed frequencies match expected frequencies.

**Test Statistic**:
χ² = Σ(O_i - E_i)²/E_i

**Example**: Test if die is fair
- Observed: (1:8, 2:12, 3:10, 4:9, 5:11, 6:10)
- Expected: (1:10, 2:10, 3:10, 4:10, 5:10, 6:10)
- χ² = (8-10)²/10 + (12-10)²/10 + ... + (10-10)²/10
- χ² = 4/10 + 4/10 + 0/10 + 1/10 + 1/10 + 0/10 = 1.0
- df = 5, p-value ≈ 0.96
- Since p > 0.05, fail to reject H₀ (die appears fair)

### Chi-Square Test of Independence
Tests whether two categorical variables are independent.

**Test Statistic**:
χ² = Σ(O_ij - E_ij)²/E_ij

**Example**: Test if gender and preference are independent
- Observed frequencies:
  - Male: (Coffee: 20, Tea: 15, Water: 10)
  - Female: (Coffee: 25, Tea: 20, Water: 15)
- Calculate expected frequencies using row and column totals
- χ² = Σ(O_ij - E_ij)²/E_ij
- df = (rows-1)(columns-1) = (2-1)(3-1) = 2
- Compare to χ² critical value

## Choosing the Right Test

### For Means
- **One sample**: One-sample t-test or z-test
- **Two independent samples**: Two-sample t-test
- **Paired samples**: Paired t-test
- **More than two groups**: ANOVA (covered in next tutorial)

### For Proportions
- **One sample**: One-proportion z-test
- **Two samples**: Two-proportion z-test

### For Categorical Data
- **Goodness-of-fit**: Chi-square goodness-of-fit test
- **Independence**: Chi-square test of independence

### Assumptions to Check
1. **Normality**: Use normal probability plots or Shapiro-Wilk test
2. **Independence**: Ensure random sampling
3. **Equal variances**: Use Levene's test for two-sample t-test
4. **Sample size**: Ensure adequate sample sizes

## Effect Size

### Cohen's d
Measures standardized difference between means:
d = (x̄₁ - x̄₂)/s_pooled

**Interpretation**:
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect

### Practical Significance
- Statistical significance ≠ practical significance
- Consider effect size and confidence intervals
- Large samples can detect tiny differences

## Common Mistakes

### 1. Misinterpreting P-values
- P-value is NOT probability that H₀ is true
- P-value is probability of data given H₀ is true
- Don't say "probability of being wrong"

### 2. Multiple Testing
- Testing multiple hypotheses increases Type I error
- Use Bonferroni correction: α_adjusted = α/k
- Consider family-wise error rate

### 3. Data Dredging
- Don't test every possible hypothesis
- Formulate hypotheses before seeing data
- Use exploratory vs. confirmatory analysis

### 4. Ignoring Assumptions
- Check normality, independence, equal variances
- Use appropriate tests for your data
- Consider nonparametric alternatives

## Practice Problems

### Problem 1
A company claims their batteries last 100 hours. Test this claim with sample data: n = 25, x̄ = 98, s = 8. Use α = 0.05.

**Solution**:
- H₀: μ = 100, H₁: μ ≠ 100
- t = (98 - 100)/(8/√25) = -2/1.6 = -1.25
- df = 24, p-value ≈ 0.22
- Since p > 0.05, fail to reject H₀
- No evidence that batteries don't last 100 hours

### Problem 2
Compare effectiveness of two drugs. Drug A: n₁ = 20, x̄₁ = 85, s₁ = 10. Drug B: n₂ = 18, x̄₂ = 90, s₂ = 12. Use α = 0.05.

**Solution**:
- H₀: μ₁ = μ₂, H₁: μ₁ ≠ μ₂
- SE = √(100/20 + 144/18) = √(5 + 8) = √13 = 3.61
- t = (85 - 90)/3.61 = -1.38
- df ≈ 35, p-value ≈ 0.18
- Since p > 0.05, fail to reject H₀
- No significant difference between drugs

### Problem 3
Test if coin is fair with 60 heads in 100 tosses. Use α = 0.05.

**Solution**:
- H₀: p = 0.5, H₁: p ≠ 0.5
- z = (0.6 - 0.5)/√(0.5×0.5/100) = 0.1/0.05 = 2.0
- p-value = 2×P(Z > 2.0) = 2×0.0228 = 0.0456
- Since p < 0.05, reject H₀
- Evidence that coin is not fair

## Key Takeaways
- Hypothesis testing provides framework for statistical decision making
- Null hypothesis is assumed true until evidence suggests otherwise
- P-values measure strength of evidence against null hypothesis
- Type I and Type II errors are important considerations
- Choose appropriate test based on data type and assumptions
- Always check assumptions before conducting tests
- Consider effect size, not just statistical significance

## Next Steps
In the next tutorial, we'll explore confidence intervals, learning how to estimate population parameters with specified levels of confidence and understand the relationship between hypothesis testing and interval estimation.
