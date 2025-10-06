# Linear Algebra Tutorial 03: Systems of Linear Equations

## Learning Objectives
By the end of this tutorial, you will be able to:
- Solve systems using Gaussian elimination
- Apply Gauss-Jordan elimination
- Determine consistency and inconsistency
- Solve homogeneous and non-homogeneous systems
- Apply systems to real-world problems

## Introduction to Systems

### Definition
A system of linear equations is a collection of equations with the same variables.

**General form**:
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ

### Matrix Form
Ax = b, where:
- A is the coefficient matrix
- x is the variable vector
- b is the constant vector

## Gaussian Elimination

### Process
1. Write augmented matrix [A | b]
2. Use elementary row operations to get row echelon form
3. Use back substitution to solve

### Elementary Row Operations
1. **Row Swap**: Rᵢ ↔ Rⱼ
2. **Row Scaling**: Rᵢ → cRᵢ (c ≠ 0)
3. **Row Addition**: Rᵢ → Rᵢ + cRⱼ

### Example
Solve:
x + 2y + z = 3
2x + 5y + 3z = 8
x + 4y + 2z = 7

**Solution**:
1. Augmented matrix: [1  2  1 | 3]
                   [2  5  3 | 8]
                   [1  4  2 | 7]

2. R₂ - 2R₁: [1  2  1 | 3]
             [0  1  1 | 2]
             [1  4  2 | 7]

3. R₃ - R₁: [1  2  1 | 3]
            [0  1  1 | 2]
            [0  2  1 | 4]

4. R₃ - 2R₂: [1  2  1 | 3]
             [0  1  1 | 2]
             [0  0 -1 | 0]

5. R₃ → -R₃: [1  2  1 | 3]
             [0  1  1 | 2]
             [0  0  1 | 0]

6. Back substitution:
   - From R₃: z = 0
   - From R₂: y + z = 2 → y = 2
   - From R₁: x + 2y + z = 3 → x = -1

## Gauss-Jordan Elimination

### Process
Continue Gaussian elimination to get reduced row echelon form (RREF).

### Example
Continue from previous example:

1. R₂ - R₃: [1  2  1 | 3]
            [0  1  0 | 2]
            [0  0  1 | 0]

2. R₁ - R₃: [1  2  0 | 3]
            [0  1  0 | 2]
            [0  0  1 | 0]

3. R₁ - 2R₂: [1  0  0 | -1]
             [0  1  0 |  2]
             [0  0  1 |  0]

Solution: x = -1, y = 2, z = 0

## Consistency and Inconsistency

### Consistent Systems
- **Unique solution**: Rank(A) = Rank([A|b]) = n
- **Infinitely many solutions**: Rank(A) = Rank([A|b]) < n

### Inconsistent Systems
- **No solution**: Rank(A) < Rank([A|b])

### Example - Inconsistent System
x + y = 1
x + y = 2

Augmented matrix: [1  1 | 1]
                  [1  1 | 2]

R₂ - R₁: [1  1 | 1]
         [0  0 | 1]

The last row represents 0 = 1, which is impossible. No solution.

### Example - Infinitely Many Solutions
x + y = 1
2x + 2y = 2

Augmented matrix: [1  1 | 1]
                  [2  2 | 2]

R₂ - 2R₁: [1  1 | 1]
          [0  0 | 0]

Solution: x = 1 - t, y = t (where t is any real number)

## Homogeneous Systems

### Definition
A homogeneous system has the form Ax = 0 (all constants are zero).

### Properties
1. Always has at least the trivial solution x = 0
2. Has non-trivial solutions if and only if det(A) = 0
3. If it has non-trivial solutions, it has infinitely many

### Example
x + 2y + z = 0
2x + 4y + 2z = 0
x + 2y + z = 0

Augmented matrix: [1  2  1 | 0]
                  [2  4  2 | 0]
                  [1  2  1 | 0]

R₂ - 2R₁: [1  2  1 | 0]
          [0  0  0 | 0]
          [1  2  1 | 0]

R₃ - R₁: [1  2  1 | 0]
         [0  0  0 | 0]
         [0  0  0 | 0]

Solution: x = -2s - t, y = s, z = t (where s, t are any real numbers)

## Applications

### Network Analysis
**Example**: Traffic flow at intersections
At intersection A: x₁ + x₂ = 100 (incoming = outgoing)
At intersection B: x₂ + x₃ = 150
At intersection C: x₁ + x₃ = 120

System:
x₁ + x₂ = 100
x₂ + x₃ = 150
x₁ + x₃ = 120

Solution: x₁ = 35, x₂ = 65, x₃ = 85

### Economics
**Example**: Input-output model
Three industries: Agriculture (A), Manufacturing (M), Services (S)

A produces: 0.2A + 0.1M + 0.1S + 50 (external demand)
M produces: 0.3A + 0.2M + 0.2S + 100
S produces: 0.1A + 0.3M + 0.1S + 80

System:
A = 0.2A + 0.1M + 0.1S + 50
M = 0.3A + 0.2M + 0.2S + 100
S = 0.1A + 0.3M + 0.1S + 80

Rearranged:
0.8A - 0.1M - 0.1S = 50
-0.3A + 0.8M - 0.2S = 100
-0.1A - 0.3M + 0.9S = 80

## Practice Problems

### Problem 1
Solve using Gaussian elimination:
2x + 3y - z = 1
x - y + 2z = 3
3x + 2y + z = 2

**Solution**:
Augmented matrix: [2  3 -1 | 1]
                  [1 -1  2 | 3]
                  [3  2  1 | 2]

R₁ ↔ R₂: [1 -1  2 | 3]
          [2  3 -1 | 1]
          [3  2  1 | 2]

R₂ - 2R₁: [1 -1  2 | 3]
          [0  5 -5 | -5]
          [3  2  1 | 2]

R₃ - 3R₁: [1 -1  2 | 3]
          [0  5 -5 | -5]
          [0  5 -5 | -7]

R₃ - R₂: [1 -1  2 | 3]
         [0  5 -5 | -5]
         [0  0  0 | -2]

Inconsistent system (no solution)

### Problem 2
Solve the homogeneous system:
x + y - z = 0
2x - y + 3z = 0
x + 2y - 2z = 0

**Solution**:
Augmented matrix: [1  1 -1 | 0]
                  [2 -1  3 | 0]
                  [1  2 -2 | 0]

R₂ - 2R₁: [1  1 -1 | 0]
          [0 -3  5 | 0]
          [1  2 -2 | 0]

R₃ - R₁: [1  1 -1 | 0]
         [0 -3  5 | 0]
         [0  1 -1 | 0]

R₂ ↔ R₃: [1  1 -1 | 0]
          [0  1 -1 | 0]
          [0 -3  5 | 0]

R₃ + 3R₂: [1  1 -1 | 0]
          [0  1 -1 | 0]
          [0  0  2 | 0]

R₃ → (1/2)R₃: [1  1 -1 | 0]
              [0  1 -1 | 0]
              [0  0  1 | 0]

Back substitution: z = 0, y = 0, x = 0
Only trivial solution

### Problem 3
Find all solutions to:
x + 2y + 3z = 6
2x + 4y + 6z = 12

**Solution**:
Augmented matrix: [1  2  3 | 6]
                  [2  4  6 | 12]

R₂ - 2R₁: [1  2  3 | 6]
          [0  0  0 | 0]

Solution: x = 6 - 2s - 3t, y = s, z = t
(Infinitely many solutions)

## Key Takeaways
- Gaussian elimination systematically solves linear systems
- Gauss-Jordan elimination produces reduced row echelon form
- Consistency depends on ranks of coefficient and augmented matrices
- Homogeneous systems always have the trivial solution
- Systems have applications in many fields

## Next Steps
In the next tutorial, we'll explore determinants in detail, learning about their properties, computation methods, and applications including Cramer's rule.
