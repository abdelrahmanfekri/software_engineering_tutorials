# Statistics Tutorial 06: Analysis of Variance (ANOVA)

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the F-test and its applications
- Perform one-way ANOVA for multiple group comparisons
- Conduct two-way ANOVA with interactions
- Interpret ANOVA results and F-statistics
- Perform post-hoc comparisons
- Check ANOVA assumptions
- Understand the relationship between ANOVA and regression

## Introduction to ANOVA

### What is ANOVA?
Analysis of Variance (ANOVA) is a statistical method used to compare means across multiple groups. It tests whether there are statistically significant differences between group means.

### Why Use ANOVA Instead of Multiple t-tests?
- **Multiple comparisons problem**: Increases Type I error rate
- **Efficiency**: Single test instead of multiple tests
- **Power**: More powerful than multiple t-tests
- **Overall significance**: Tests if any groups differ

### The F-Test
ANOVA uses the F-distribution to test hypotheses about group means.

**F-statistic**:
F = MS_between / MS_within

Where:
- MS_between = Mean Square Between Groups
- MS_within = Mean Square Within Groups

## One-Way ANOVA

### Model
y_ij = μ + α_i + ε_ij

Where:
- y_ij = observation j in group i
- μ = overall mean
- α_i = effect of group i
- ε_ij = random error

### Hypotheses
- H₀: μ₁ = μ₂ = ... = μ_k (all group means equal)
- H₁: At least one group mean differs

### Sum of Squares Decomposition
**Total Sum of Squares (SST)**:
SST = ΣΣ(y_ij - ȳ)²

**Between Groups Sum of Squares (SSB)**:
SSB = Σn_i(ȳ_i - ȳ)²

**Within Groups Sum of Squares (SSW)**:
SSW = ΣΣ(y_ij - ȳ_i)²

**Relationship**: SST = SSB + SSW

### Degrees of Freedom
- df_between = k - 1 (number of groups - 1)
- df_within = N - k (total observations - number of groups)
- df_total = N - 1 (total observations - 1)

### Mean Squares
- MS_between = SSB / df_between
- MS_within = SSW / df_within

### F-Statistic
F = MS_between / MS_within

**Example**: Compare test scores across three teaching methods
- Method A: [78, 82, 85, 79, 81] (n₁ = 5, ȳ₁ = 81)
- Method B: [75, 77, 80, 76, 78] (n₂ = 5, ȳ₂ = 77.2)
- Method C: [85, 87, 90, 86, 88] (n₃ = 5, ȳ₃ = 87.2)
- Overall mean: ȳ = 81.8

**Calculations**:
- SSB = 5(81-81.8)² + 5(77.2-81.8)² + 5(87.2-81.8)²
- SSB = 5(0.64) + 5(21.16) + 5(29.16) = 3.2 + 105.8 + 145.8 = 254.8
- SSW = Σ(y_ij - ȳ_i)² = 20 + 12.8 + 20 = 52.8
- MS_between = 254.8/2 = 127.4
- MS_within = 52.8/12 = 4.4
- F = 127.4/4.4 = 28.95

### ANOVA Table
| Source | SS | df | MS | F | p-value |
|--------|----|----|----|----|---------|
| Between | 254.8 | 2 | 127.4 | 28.95 | < 0.001 |
| Within | 52.8 | 12 | 4.4 | | |
| Total | 307.6 | 14 | | | |

## ANOVA Assumptions

### 1. Independence
- Observations are independent
- Random sampling
- No repeated measures (unless accounted for)

### 2. Normality
- Data in each group is normally distributed
- Check with normal probability plots
- Use Shapiro-Wilk test

### 3. Homoscedasticity (Equal Variances)
- All groups have equal variances
- Check with Levene's test
- Use Bartlett's test

### 4. Random Sampling
- Groups are randomly selected
- Representative samples

### Checking Assumptions
**Normality**: Normal probability plots, Shapiro-Wilk test
**Equal Variances**: Levene's test, Bartlett's test
**Independence**: Study design, residual plots

## Post-Hoc Comparisons

### When to Use Post-Hoc Tests
- ANOVA shows significant differences
- Need to identify which groups differ
- Multiple pairwise comparisons

### Common Post-Hoc Tests

#### Tukey's HSD (Honestly Significant Difference)
**Formula**: HSD = q_(α,k,df) × √(MS_within/n)

**Example**: From previous example
- q_(0.05,3,12) = 3.77
- HSD = 3.77 × √(4.4/5) = 3.77 × 0.94 = 3.54
- Compare all pairwise differences to 3.54

#### Bonferroni Correction
**Method**: Divide α by number of comparisons
- 3 groups: 3 comparisons, α_adjusted = 0.05/3 = 0.0167
- Use t-tests with adjusted α

#### Scheffé's Test
**Formula**: S = √((k-1)F_(α,k-1,N-k)) × √(2MS_within/n)

### Example Post-Hoc Results
| Comparison | Difference | HSD | Significant? |
|------------|------------|-----|--------------|
| A vs B | 3.8 | 3.54 | Yes |
| A vs C | -6.2 | 3.54 | Yes |
| B vs C | -10.0 | 3.54 | Yes |

## Two-Way ANOVA

### Model
y_ijk = μ + α_i + β_j + (αβ)_ij + ε_ijk

Where:
- α_i = effect of factor A (level i)
- β_j = effect of factor B (level j)
- (αβ)_ij = interaction effect

### Types of Effects
1. **Main Effects**: Individual factor effects
2. **Interaction Effects**: Combined factor effects
3. **Simple Effects**: Effect of one factor at specific level of other

### Example: Two-Way ANOVA
**Factors**: Teaching Method (A, B, C) × Class Size (Small, Large)
**Response**: Test Scores

| Method | Small Class | Large Class | Row Mean |
|--------|-------------|-------------|----------|
| A | 85, 87 | 78, 80 | 82.5 |
| B | 82, 84 | 75, 77 | 79.5 |
| C | 90, 92 | 85, 87 | 88.5 |
| Column Mean | 86.5 | 81.0 | 83.75 |

### Two-Way ANOVA Table
| Source | SS | df | MS | F | p-value |
|--------|----|----|----|----|---------|
| Method | 162 | 2 | 81 | 20.25 | < 0.001 |
| Class Size | 60.75 | 1 | 60.75 | 15.19 | 0.001 |
| Interaction | 0.75 | 2 | 0.375 | 0.094 | 0.91 |
| Error | 48 | 12 | 4 | | |
| Total | 271.5 | 17 | | | |

### Interpreting Interactions
- **No Interaction**: Effect of one factor is same at all levels of other factor
- **Interaction Present**: Effect of one factor depends on level of other factor

## Effect Size

### Eta-Squared (η²)
η² = SS_effect / SS_total

**Interpretation**:
- η² = 0.01: Small effect
- η² = 0.06: Medium effect
- η² = 0.14: Large effect

### Partial Eta-Squared (η²_p)
η²_p = SS_effect / (SS_effect + SS_error)

### Example
From one-way ANOVA:
- η² = 254.8/307.6 = 0.83 (large effect)
- Teaching method explains 83% of variance in test scores

## ANOVA vs. Regression

### Relationship
- ANOVA is special case of regression
- Categorical predictors in regression
- Dummy coding for groups

### Regression Approach
**Model**: y = β₀ + β₁D₁ + β₂D₂ + ε

Where D₁, D₂ are dummy variables for groups.

### Advantages of Each Approach
**ANOVA**:
- Easier interpretation
- Built-in post-hoc tests
- Clear group comparisons

**Regression**:
- More flexible
- Can include continuous predictors
- Easier to extend to multiple factors

## Common Mistakes

### 1. Multiple t-tests Instead of ANOVA
- Increases Type I error rate
- Less powerful
- No overall test of significance

### 2. Ignoring Assumptions
- Check normality and equal variances
- Use transformations if needed
- Consider nonparametric alternatives

### 3. Not Using Post-Hoc Tests
- ANOVA only tells if groups differ
- Need post-hoc tests to identify which groups
- Choose appropriate post-hoc method

### 4. Misinterpreting Interactions
- Don't interpret main effects when interaction present
- Plot interaction effects
- Consider simple effects

## Practice Problems

### Problem 1
Compare three diets for weight loss:
- Diet A: [5, 7, 6, 8, 4] pounds lost
- Diet B: [3, 4, 5, 6, 2] pounds lost
- Diet C: [8, 9, 10, 7, 9] pounds lost

**Solution**:
- ȳ_A = 6, ȳ_B = 4, ȳ_C = 8.6, ȳ = 6.2
- SSB = 5(6-6.2)² + 5(4-6.2)² + 5(8.6-6.2)² = 0.2 + 24.2 + 28.8 = 53.2
- SSW = 10 + 10 + 8.8 = 28.8
- MS_between = 53.2/2 = 26.6
- MS_within = 28.8/12 = 2.4
- F = 26.6/2.4 = 11.08
- p-value < 0.01, reject H₀

### Problem 2
For Problem 1, perform Tukey's HSD test.

**Solution**:
- q_(0.05,3,12) = 3.77
- HSD = 3.77 × √(2.4/5) = 3.77 × 0.69 = 2.60
- A vs B: |6-4| = 2 < 2.60, not significant
- A vs C: |6-8.6| = 2.6 = 2.60, significant
- B vs C: |4-8.6| = 4.6 > 2.60, significant

### Problem 3
Calculate effect size for Problem 1.

**Solution**:
- η² = 53.2/(53.2 + 28.8) = 53.2/82 = 0.65
- Large effect size

## Key Takeaways
- ANOVA compares means across multiple groups
- F-test determines if group means differ significantly
- Check assumptions before conducting ANOVA
- Use post-hoc tests to identify which groups differ
- Two-way ANOVA includes interaction effects
- Effect size measures practical significance
- ANOVA is related to regression analysis

## Next Steps
In the next tutorial, we'll explore nonparametric statistics, learning about alternatives to parametric tests when assumptions are violated or when dealing with ordinal data.
