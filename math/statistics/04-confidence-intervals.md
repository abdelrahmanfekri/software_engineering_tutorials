# Statistics Tutorial 04: Confidence Intervals

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand point vs. interval estimation
- Calculate confidence intervals for means and proportions
- Interpret confidence intervals correctly
- Determine required sample sizes
- Understand margin of error concepts
- Apply confidence intervals in real-world scenarios

## Introduction to Confidence Intervals

### What are Confidence Intervals?
A confidence interval is a range of values that likely contains the true population parameter. It provides both an estimate and a measure of uncertainty about that estimate.

### Point vs. Interval Estimation
- **Point Estimate**: Single value estimate (e.g., sample mean)
- **Interval Estimate**: Range of values (e.g., confidence interval)
- **Advantage of Intervals**: Provide measure of precision

### Confidence Level
- Probability that interval contains true parameter
- Common levels: 90%, 95%, 99%
- Higher confidence → wider interval
- Lower confidence → narrower interval

## Confidence Intervals for Means

### When Population Standard Deviation is Known (z-interval)

**Formula**:
CI = x̄ ± z_(α/2) × (σ/√n)

**Example**: Estimate average height with σ = 3 inches
- Sample: n = 36, x̄ = 70 inches
- 95% confidence: z_(0.025) = 1.96
- CI = 70 ± 1.96 × (3/√36) = 70 ± 1.96 × 0.5 = 70 ± 0.98
- 95% CI: (69.02, 70.98)

### When Population Standard Deviation is Unknown (t-interval)

**Formula**:
CI = x̄ ± t_(α/2, df) × (s/√n)

**Example**: Estimate average test score
- Sample: n = 25, x̄ = 78, s = 8
- 95% confidence: df = 24, t_(0.025, 24) = 2.064
- CI = 78 ± 2.064 × (8/√25) = 78 ± 2.064 × 1.6 = 78 ± 3.30
- 95% CI: (74.70, 81.30)

### Factors Affecting Width
1. **Sample size**: Larger n → narrower interval
2. **Standard deviation**: Larger s → wider interval
3. **Confidence level**: Higher confidence → wider interval

## Confidence Intervals for Proportions

### Large Sample Approximation

**Formula**:
CI = p̂ ± z_(α/2) × √(p̂(1-p̂)/n)

**Example**: Estimate proportion of voters supporting candidate
- Sample: n = 400, p̂ = 0.55
- 95% confidence: z_(0.025) = 1.96
- CI = 0.55 ± 1.96 × √(0.55×0.45/400)
- CI = 0.55 ± 1.96 × √(0.2475/400) = 0.55 ± 1.96 × 0.0249
- CI = 0.55 ± 0.0488
- 95% CI: (0.501, 0.599)

### Small Sample Correction
When np̂ < 5 or n(1-p̂) < 5, use exact methods or continuity correction.

### Continuity Correction
Add ±0.5/n to interval endpoints for better approximation.

## Margin of Error

### Definition
Margin of error is half the width of the confidence interval.

**For means**: ME = z_(α/2) × (σ/√n)
**For proportions**: ME = z_(α/2) × √(p̂(1-p̂)/n)

### Example
- 95% CI for mean: (45, 55)
- Margin of error = (55 - 45)/2 = 5
- Point estimate = 50

### Factors Affecting Margin of Error
1. **Confidence level**: Higher confidence → larger ME
2. **Sample size**: Larger n → smaller ME
3. **Population variability**: Larger σ → larger ME

## Sample Size Determination

### For Estimating Means
n = (z_(α/2) × σ/E)²

**Example**: Want 95% confidence, σ = 10, margin of error = 2
- n = (1.96 × 10/2)² = (9.8)² = 96.04 ≈ 97

### For Estimating Proportions
n = (z_(α/2)/E)² × p̂(1-p̂)

**Conservative approach** (when p̂ unknown):
n = (z_(α/2)/E)² × 0.25

**Example**: Want 95% confidence, margin of error = 0.03
- Conservative: n = (1.96/0.03)² × 0.25 = (65.33)² × 0.25 = 1067

### Finite Population Correction
When sampling from finite population:
n_adjusted = n/(1 + (n-1)/N)

## Interpretation of Confidence Intervals

### Correct Interpretation
"We are 95% confident that the true population mean lies between [lower bound] and [upper bound]."

### Incorrect Interpretations
- "There's a 95% probability that the true mean is in this interval"
- "95% of the data falls in this interval"
- "The true mean is definitely in this interval"

### What Confidence Level Means
- If we repeated sampling 100 times
- About 95 intervals would contain true parameter
- About 5 intervals would not contain true parameter

## One-Sided Confidence Intervals

### Lower Bound
CI = [x̄ - t_(α, df) × (s/√n), ∞)

### Upper Bound
CI = (-∞, x̄ + t_(α, df) × (s/√n)]

**Example**: Want 95% upper bound for mean
- Sample: n = 20, x̄ = 50, s = 5
- df = 19, t_(0.05, 19) = 1.729
- Upper bound = 50 + 1.729 × (5/√20) = 50 + 1.93 = 51.93
- 95% upper bound: (-∞, 51.93]

## Confidence Intervals for Differences

### Difference of Means (Independent Samples)
CI = (x̄₁ - x̄₂) ± t_(α/2, df) × SE(x̄₁ - x̄₂)

Where SE(x̄₁ - x̄₂) = √(s₁²/n₁ + s₂²/n₂)

**Example**: Compare two teaching methods
- Method A: n₁ = 30, x̄₁ = 78, s₁ = 8
- Method B: n₂ = 25, x̄₂ = 82, s₂ = 7
- Difference = 78 - 82 = -4
- SE = √(64/30 + 49/25) = √(2.13 + 1.96) = 2.02
- df ≈ 53, t_(0.025, 53) ≈ 2.006
- CI = -4 ± 2.006 × 2.02 = -4 ± 4.05
- 95% CI: (-8.05, 0.05)

### Difference of Proportions
CI = (p̂₁ - p̂₂) ± z_(α/2) × √(p̂₁(1-p̂₁)/n₁ + p̂₂(1-p̂₂)/n₂)

## Bootstrap Confidence Intervals

### Method
1. Resample with replacement from original sample
2. Calculate statistic for each bootstrap sample
3. Repeat many times (e.g., 1000)
4. Use percentiles of bootstrap distribution

### Advantages
- No distributional assumptions
- Works for any statistic
- Handles complex sampling designs

### Example
- Original sample: [2, 4, 6, 8, 10]
- Bootstrap samples: [4, 2, 8, 4, 6], [6, 10, 2, 8, 4], ...
- Calculate mean for each bootstrap sample
- 95% CI = (2.5th percentile, 97.5th percentile)

## Common Mistakes

### 1. Misinterpreting Confidence Level
- Don't say "probability that parameter is in interval"
- Say "confidence that interval contains parameter"

### 2. Ignoring Assumptions
- Check normality for small samples
- Ensure random sampling
- Verify independence

### 3. Multiple Comparisons
- Multiple intervals increase overall error rate
- Use Bonferroni correction if needed

### 4. Sample Size Issues
- Too small samples give wide intervals
- Plan adequate sample sizes

## Practice Problems

### Problem 1
A sample of 50 students has mean GPA 3.2 with standard deviation 0.5. Find 95% confidence interval for population mean GPA.

**Solution**:
- n = 50, x̄ = 3.2, s = 0.5
- df = 49, t_(0.025, 49) ≈ 2.009
- CI = 3.2 ± 2.009 × (0.5/√50)
- CI = 3.2 ± 2.009 × 0.0707 = 3.2 ± 0.142
- 95% CI: (3.058, 3.342)

### Problem 2
In a survey of 200 people, 120 support a policy. Find 90% confidence interval for population proportion.

**Solution**:
- n = 200, p̂ = 120/200 = 0.6
- 90% confidence: z_(0.05) = 1.645
- CI = 0.6 ± 1.645 × √(0.6×0.4/200)
- CI = 0.6 ± 1.645 × √(0.24/200) = 0.6 ± 1.645 × 0.0346
- CI = 0.6 ± 0.057
- 90% CI: (0.543, 0.657)

### Problem 3
How large a sample needed to estimate mean within 2 units with 95% confidence, given σ = 8?

**Solution**:
- n = (z_(α/2) × σ/E)² = (1.96 × 8/2)² = (7.84)² = 61.47 ≈ 62

### Problem 4
Compare two groups: Group A (n=30, x̄=45, s=6) vs Group B (n=25, x̄=48, s=5). Find 95% CI for difference in means.

**Solution**:
- Difference = 45 - 48 = -3
- SE = √(36/30 + 25/25) = √(1.2 + 1.0) = √2.2 = 1.48
- df ≈ 53, t_(0.025, 53) ≈ 2.006
- CI = -3 ± 2.006 × 1.48 = -3 ± 2.97
- 95% CI: (-5.97, -0.03)

## Key Takeaways
- Confidence intervals provide both estimate and measure of uncertainty
- Higher confidence levels result in wider intervals
- Larger sample sizes result in narrower intervals
- Correct interpretation is crucial for proper use
- Sample size planning ensures adequate precision
- Bootstrap methods provide alternatives when assumptions fail

## Next Steps
In the next tutorial, we'll explore correlation and regression analysis, learning how to measure relationships between variables and build predictive models using linear regression techniques.
