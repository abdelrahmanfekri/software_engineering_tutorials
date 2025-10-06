# College Algebra Tutorial 05: Sequences and Series

## Learning Objectives
By the end of this tutorial, you will be able to:
- Identify and work with arithmetic sequences
- Identify and work with geometric sequences
- Find sums of arithmetic and geometric series
- Use summation notation
- Apply sequences and series to real-world problems
- Understand infinite series and convergence

## Introduction to Sequences

### Definition
A sequence is an ordered list of numbers. Each number in the sequence is called a term.

**Notation**: {aₙ} or a₁, a₂, a₃, ..., aₙ

**Example**: 2, 5, 8, 11, 14, ... is a sequence where a₁ = 2, a₂ = 5, etc.

### Types of Sequences
1. **Arithmetic**: Each term differs from the previous by a constant
2. **Geometric**: Each term is multiplied by a constant to get the next
3. **Fibonacci**: Each term is the sum of the two previous terms
4. **Other**: Various patterns and rules

## Arithmetic Sequences

### Definition
An arithmetic sequence has the form: aₙ = a₁ + (n - 1)d
Where:
- a₁ = first term
- d = common difference
- n = term number

### Finding Terms
**Example**: Find the 10th term of 3, 7, 11, 15, ...
- a₁ = 3, d = 4
- a₁₀ = 3 + (10 - 1)(4) = 3 + 9(4) = 3 + 36 = 39

### Finding the Common Difference
d = aₙ₊₁ - aₙ for any n

**Example**: In 5, 8, 11, 14, ...
- d = 8 - 5 = 3 (or 11 - 8 = 3, etc.)

## Arithmetic Series

### Definition
The sum of terms in an arithmetic sequence.

### Formula for Sum
Sₙ = n(a₁ + aₙ)/2
or
Sₙ = n[2a₁ + (n - 1)d]/2

**Example**: Find the sum of the first 20 terms of 2, 5, 8, 11, ...
- a₁ = 2, d = 3, n = 20
- a₂₀ = 2 + (20 - 1)(3) = 2 + 19(3) = 59
- S₂₀ = 20(2 + 59)/2 = 20(61)/2 = 610

### Summation Notation
The sum can be written as: Σ(k=1 to n) aₖ

**Example**: Σ(k=1 to 5) (2k + 1) = 3 + 5 + 7 + 9 + 11 = 35

## Geometric Sequences

### Definition
A geometric sequence has the form: aₙ = a₁ · r^(n-1)
Where:
- a₁ = first term
- r = common ratio
- n = term number

### Finding Terms
**Example**: Find the 8th term of 2, 6, 18, 54, ...
- a₁ = 2, r = 3
- a₈ = 2 · 3^(8-1) = 2 · 3⁷ = 2 · 2187 = 4374

### Finding the Common Ratio
r = aₙ₊₁/aₙ for any n

**Example**: In 4, 12, 36, 108, ...
- r = 12/4 = 3 (or 36/12 = 3, etc.)

## Geometric Series

### Definition
The sum of terms in a geometric sequence.

### Formula for Sum
Sₙ = a₁(1 - rⁿ)/(1 - r) if r ≠ 1
Sₙ = na₁ if r = 1

**Example**: Find the sum of the first 6 terms of 3, 6, 12, 24, ...
- a₁ = 3, r = 2, n = 6
- S₆ = 3(1 - 2⁶)/(1 - 2) = 3(1 - 64)/(-1) = 3(-63)/(-1) = 189

### Infinite Geometric Series
If |r| < 1, the infinite sum converges to:
S = a₁/(1 - r)

**Example**: Find the sum of 1 + 1/2 + 1/4 + 1/8 + ...
- a₁ = 1, r = 1/2
- S = 1/(1 - 1/2) = 1/(1/2) = 2

## Applications

### Compound Interest
A = P(1 + r)^t
Where:
- A = final amount
- P = principal
- r = interest rate per period
- t = number of periods

**Example**: $1000 invested at 5% annually for 10 years
- A = 1000(1.05)^10 ≈ $1628.89

### Population Growth
P(t) = P₀(1 + r)^t
Where:
- P(t) = population at time t
- P₀ = initial population
- r = growth rate
- t = time

**Example**: Population of 1000 grows at 3% annually
- P(5) = 1000(1.03)^5 ≈ 1159

### Annuities
Future value of regular payments:
FV = PMT[(1 + r)^n - 1]/r
Where:
- PMT = payment amount
- r = interest rate per period
- n = number of payments

**Example**: $100 monthly payments for 5 years at 6% annual interest
- r = 0.06/12 = 0.005, n = 60
- FV = 100[(1.005)^60 - 1]/0.005 ≈ $6977

## Special Sequences

### Fibonacci Sequence
1, 1, 2, 3, 5, 8, 13, 21, ...
Each term is the sum of the two previous terms.

### Triangular Numbers
1, 3, 6, 10, 15, 21, ...
Formula: Tₙ = n(n + 1)/2

### Square Numbers
1, 4, 9, 16, 25, 36, ...
Formula: Sₙ = n²

## Practice Problems

### Problem 1
Find the 15th term and sum of first 15 terms of the arithmetic sequence: 4, 9, 14, 19, ...

**Solution**:
- a₁ = 4, d = 5
- a₁₅ = 4 + (15 - 1)(5) = 4 + 14(5) = 74
- S₁₅ = 15(4 + 74)/2 = 15(78)/2 = 585

### Problem 2
Find the 7th term and sum of first 7 terms of the geometric sequence: 2, 6, 18, 54, ...

**Solution**:
- a₁ = 2, r = 3
- a₇ = 2 · 3^(7-1) = 2 · 3⁶ = 2 · 729 = 1458
- S₇ = 2(1 - 3⁷)/(1 - 3) = 2(1 - 2187)/(-2) = 2(-2186)/(-2) = 2186

### Problem 3
A ball is dropped from 100 feet. Each bounce reaches 2/3 of the previous height. Find the total distance traveled.

**Solution**:
- Initial drop: 100 feet
- First bounce up: 100(2/3) = 200/3 feet
- First bounce down: 200/3 feet
- Second bounce up: 100(2/3)² = 400/9 feet
- Second bounce down: 400/9 feet
- Pattern: 100 + 2(200/3) + 2(400/9) + 2(800/27) + ...
- Total = 100 + 2(200/3)/(1 - 2/3) = 100 + 2(200/3)/(1/3) = 100 + 400 = 500 feet

### Problem 4
Find the sum: Σ(k=1 to 10) (3k - 2)

**Solution**:
This is an arithmetic series with a₁ = 1, d = 3, n = 10
- a₁₀ = 1 + (10 - 1)(3) = 1 + 27 = 28
- S₁₀ = 10(1 + 28)/2 = 10(29)/2 = 145

## Key Takeaways
- Arithmetic sequences have constant differences
- Geometric sequences have constant ratios
- Sum formulas exist for both finite and infinite series
- Sequences and series have many real-world applications
- Summation notation provides compact representation
- Convergence depends on the common ratio in geometric series

## Next Steps
In the next tutorial, we'll explore conic sections, learning about circles, ellipses, hyperbolas, and parabolas, and their properties and applications.
