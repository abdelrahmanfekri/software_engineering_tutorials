# Statistics Tutorial 07: Nonparametric Statistics

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand when to use nonparametric methods
- Perform Mann-Whitney U test for two independent samples
- Conduct Kruskal-Wallis test for multiple groups
- Use Wilcoxon signed-rank test for paired samples
- Apply chi-square tests for categorical data
- Choose appropriate nonparametric alternatives
- Interpret nonparametric test results

## Introduction to Nonparametric Statistics

### What are Nonparametric Tests?
Nonparametric tests are statistical methods that don't assume specific population distributions. They're based on ranks, medians, or other distribution-free statistics.

### When to Use Nonparametric Tests
1. **Data violates normality assumption**
2. **Small sample sizes**
3. **Ordinal data**
4. **Outliers present**
5. **Unequal variances**
6. **Non-normal distributions**

### Advantages
- No distributional assumptions
- Robust to outliers
- Work with ordinal data
- Valid for small samples

### Disadvantages
- Less powerful than parametric tests (when assumptions met)
- Less informative about effect size
- Limited to specific hypotheses

## Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

### Purpose
Tests whether two independent samples come from populations with the same distribution.

### Hypotheses
- H₀: Two populations have identical distributions
- H₁: Two populations have different distributions

### Method
1. **Combine samples** and rank all observations
2. **Sum ranks** for each group
3. **Calculate U statistic** for each group
4. **Use smaller U** as test statistic

### U Statistic Formula
U₁ = n₁n₂ + n₁(n₁+1)/2 - R₁
U₂ = n₁n₂ + n₂(n₂+1)/2 - R₂

Where:
- n₁, n₂ = sample sizes
- R₁, R₂ = sum of ranks for each group

### Example: Compare two teaching methods
- Method A: [78, 82, 85, 79, 81] (n₁ = 5)
- Method B: [75, 77, 80, 76, 78] (n₂ = 5)

**Ranking**:
- Combined: [75, 76, 77, 78, 78, 79, 80, 81, 82, 85]
- Ranks: [1, 2, 3, 4.5, 4.5, 6, 7, 8, 9, 10]
- Method A ranks: [6, 9, 10, 8, 4.5] → R₁ = 37.5
- Method B ranks: [1, 3, 7, 2, 4.5] → R₂ = 17.5

**U Statistics**:
- U₁ = 5×5 + 5×6/2 - 37.5 = 25 + 15 - 37.5 = 2.5
- U₂ = 5×5 + 5×6/2 - 17.5 = 25 + 15 - 17.5 = 22.5
- Test statistic: U = min(2.5, 22.5) = 2.5

### Normal Approximation (Large Samples)
z = (U - μ_U)/σ_U

Where:
- μ_U = n₁n₂/2
- σ_U = √(n₁n₂(n₁+n₂+1)/12)

### Example: z = (2.5 - 12.5)/√(25×11/12) = -10/4.79 = -2.09
- p-value ≈ 0.037 (two-tailed)

## Kruskal-Wallis Test

### Purpose
Tests whether multiple independent samples come from populations with the same distribution (nonparametric alternative to one-way ANOVA).

### Hypotheses
- H₀: All populations have identical distributions
- H₁: At least one population has different distribution

### Test Statistic
H = 12/(N(N+1)) × Σ(R_i²/n_i) - 3(N+1)

Where:
- N = total sample size
- R_i = sum of ranks for group i
- n_i = sample size for group i

### Example: Compare three diets
- Diet A: [5, 7, 6, 8, 4] (n₁ = 5)
- Diet B: [3, 4, 5, 6, 2] (n₂ = 5)
- Diet C: [8, 9, 10, 7, 9] (n₃ = 5)

**Ranking**:
- Combined: [2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]
- Ranks: [1, 2, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5, 9.5, 9.5, 11.5, 11.5, 13.5, 13.5, 15]
- Diet A ranks: [5.5, 9.5, 7.5, 11.5, 3.5] → R₁ = 37.5
- Diet B ranks: [2, 3.5, 5.5, 7.5, 1] → R₂ = 20.5
- Diet C ranks: [11.5, 13.5, 15, 9.5, 13.5] → R₃ = 64

**H Statistic**:
- H = 12/(15×16) × (37.5²/5 + 20.5²/5 + 64²/5) - 3×16
- H = 12/240 × (281.25 + 84.05 + 819.2) - 48
- H = 0.05 × 1184.5 - 48 = 59.23 - 48 = 11.23

### Critical Value
- df = k - 1 = 3 - 1 = 2
- χ²_(0.05,2) = 5.99
- Since H = 11.23 > 5.99, reject H₀

## Wilcoxon Signed-Rank Test

### Purpose
Tests whether paired observations come from populations with the same distribution (nonparametric alternative to paired t-test).

### Hypotheses
- H₀: Median difference = 0
- H₁: Median difference ≠ 0

### Method
1. **Calculate differences** between paired observations
2. **Rank absolute differences**
3. **Sum ranks** for positive and negative differences
4. **Use smaller sum** as test statistic

### Example: Test effectiveness of diet program
- Before: [180, 175, 190, 185, 170]
- After: [175, 170, 185, 180, 165]
- Differences: [-5, -5, -5, -5, -5]

**Ranking absolute differences**:
- Absolute differences: [5, 5, 5, 5, 5]
- Ranks: [3, 3, 3, 3, 3] (all tied)
- Positive ranks: 0
- Negative ranks: 15
- Test statistic: W = min(0, 15) = 0

### Normal Approximation
z = (W - μ_W)/σ_W

Where:
- μ_W = n(n+1)/4
- σ_W = √(n(n+1)(2n+1)/24)

## Chi-Square Tests

### Chi-Square Test of Independence
Tests whether two categorical variables are independent.

### Example: Test if gender and preference are independent
- Observed frequencies:
  - Male: Coffee(20), Tea(15), Water(10)
  - Female: Coffee(25), Tea(20), Water(15)

**Expected frequencies** (assuming independence):
- Total Coffee: 45, Total Tea: 35, Total Water: 25
- Total Male: 45, Total Female: 60
- Expected Male Coffee: (45×45)/105 = 19.29
- Expected Female Coffee: (60×45)/105 = 25.71

**Chi-square statistic**:
χ² = Σ(O_ij - E_ij)²/E_ij
χ² = (20-19.29)²/19.29 + (15-15)²/15 + ... + (15-14.29)²/14.29
χ² = 0.026 + 0 + 0.035 + 0.020 + 0 + 0.026 = 0.107

**Critical value**:
- df = (2-1)(3-1) = 2
- χ²_(0.05,2) = 5.99
- Since χ² = 0.107 < 5.99, fail to reject H₀

### Chi-Square Goodness-of-Fit Test
Tests whether observed frequencies match expected frequencies.

**Example**: Test if die is fair
- Observed: [8, 12, 10, 9, 11, 10]
- Expected: [10, 10, 10, 10, 10, 10]
- χ² = Σ(O_i - E_i)²/E_i = 4/10 + 4/10 + 0/10 + 1/10 + 1/10 + 0/10 = 1.0
- df = 6 - 1 = 5
- χ²_(0.05,5) = 11.07
- Since χ² = 1.0 < 11.07, fail to reject H₀

## Choosing Nonparametric Tests

### For Two Independent Samples
- **Parametric**: Two-sample t-test
- **Nonparametric**: Mann-Whitney U test

### For Multiple Independent Samples
- **Parametric**: One-way ANOVA
- **Nonparametric**: Kruskal-Wallis test

### For Paired Samples
- **Parametric**: Paired t-test
- **Nonparametric**: Wilcoxon signed-rank test

### For Categorical Data
- **Chi-square test of independence**
- **Chi-square goodness-of-fit test**

## Effect Size for Nonparametric Tests

### Rank-Biserial Correlation
For Mann-Whitney U test:
r = 1 - (2U)/(n₁n₂)

### Eta-Squared
For Kruskal-Wallis test:
η² = (H - k + 1)/(N - k)

### Example
From Mann-Whitney U example:
- r = 1 - (2×2.5)/(5×5) = 1 - 5/25 = 0.8 (large effect)

## Common Mistakes

### 1. Using Parametric Tests When Assumptions Violated
- Check normality and equal variances
- Use nonparametric alternatives when needed
- Don't ignore assumption violations

### 2. Misinterpreting Results
- Nonparametric tests test distributions, not just means
- Results may differ from parametric tests
- Consider effect size measures

### 3. Not Handling Ties Properly
- Use average ranks for tied values
- Adjust formulas for ties
- Consider continuity corrections

### 4. Ignoring Sample Size Requirements
- Some tests need minimum sample sizes
- Use exact methods for small samples
- Consider power analysis

## Practice Problems

### Problem 1
Compare two groups using Mann-Whitney U test:
- Group A: [12, 15, 18, 20, 22]
- Group B: [8, 10, 14, 16, 19]

**Solution**:
- Combined ranks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- Group A ranks: [6, 7, 8, 9, 10] → R₁ = 40
- Group B ranks: [1, 2, 4, 5, 7] → R₂ = 19
- U₁ = 5×5 + 5×6/2 - 40 = 25 + 15 - 40 = 0
- U₂ = 5×5 + 5×6/2 - 19 = 25 + 15 - 19 = 21
- U = 0, p-value < 0.01

### Problem 2
Test three groups using Kruskal-Wallis test:
- Group 1: [5, 7, 9]
- Group 2: [3, 6, 8]
- Group 3: [10, 12, 14]

**Solution**:
- Combined ranks: [1, 2, 3, 4, 5, 6, 7, 8, 9]
- Group 1 ranks: [2, 4, 6] → R₁ = 12
- Group 2 ranks: [1, 3, 5] → R₂ = 9
- Group 3 ranks: [7, 8, 9] → R₃ = 24
- H = 12/(9×10) × (12²/3 + 9²/3 + 24²/3) - 3×10
- H = 12/90 × (48 + 27 + 192) - 30 = 0.133 × 267 - 30 = 5.51
- df = 2, χ²_(0.05,2) = 5.99
- Since H = 5.51 < 5.99, fail to reject H₀

### Problem 3
Test paired data using Wilcoxon signed-rank test:
- Before: [100, 110, 95, 105, 120]
- After: [95, 105, 90, 100, 115]
- Differences: [-5, -5, -5, -5, -5]

**Solution**:
- Absolute differences: [5, 5, 5, 5, 5]
- Ranks: [3, 3, 3, 3, 3]
- Positive ranks: 0
- Negative ranks: 15
- W = 0, p-value < 0.05

## Key Takeaways
- Nonparametric tests don't assume specific distributions
- Use when parametric assumptions are violated
- Mann-Whitney U test for two independent samples
- Kruskal-Wallis test for multiple independent samples
- Wilcoxon signed-rank test for paired samples
- Chi-square tests for categorical data
- Consider effect size measures
- Choose appropriate test based on data type and assumptions

## Next Steps
In the next tutorial, we'll explore advanced statistical topics including multivariate statistics, time series analysis, and Bayesian methods, providing a foundation for more sophisticated statistical analysis.
