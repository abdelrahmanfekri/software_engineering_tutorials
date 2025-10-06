# College Algebra Tutorial 08: Matrices and Determinants

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand matrix notation and basic operations
- Calculate determinants of matrices
- Use Cramer's rule to solve systems
- Find matrix inverses
- Apply matrices to solve real-world problems

## Introduction to Matrices

### Definition
A matrix is a rectangular array of numbers arranged in rows and columns.

**Notation**: A = [aᵢⱼ] where aᵢⱼ is the element in row i, column j

**Example**: A = [2  3  1]
              [1 -1  0]
is a 2×3 matrix.

### Types of Matrices
- **Square matrix**: Same number of rows and columns
- **Row matrix**: Single row
- **Column matrix**: Single column
- **Zero matrix**: All elements are zero
- **Identity matrix**: Square matrix with 1s on diagonal, 0s elsewhere

## Matrix Operations

### Addition and Subtraction
Add/subtract corresponding elements (matrices must have same dimensions).

**Example**: [2  3] + [1 -1] = [3  2]
            [1 -1]   [0  2]   [1  1]

### Scalar Multiplication
Multiply each element by the scalar.

**Example**: 3[2  3] = [6  9]
            [1 -1]   [3 -3]

### Matrix Multiplication
For A (m×n) and B (n×p), the product AB is m×p where:
(AB)ᵢⱼ = Σ(k=1 to n) aᵢₖbₖⱼ

**Example**: [2  3][1  2] = [2(1)+3(0)  2(2)+3(1)] = [2  7]
            [1 -1][0  1]   [1(1)+(-1)(0)  1(2)+(-1)(1)] = [1  1]

### Properties
- **Associativity**: (AB)C = A(BC)
- **Distributivity**: A(B + C) = AB + AC
- **NOT Commutative**: AB ≠ BA in general

## Determinants

### 2×2 Determinant
For A = [a  b]
        [c  d]

det(A) = ad - bc

**Example**: det[2  3] = 2(1) - 3(-1) = 2 + 3 = 5
            [-1  1]

### 3×3 Determinant
Use cofactor expansion along first row:

det(A) = a₁₁C₁₁ - a₁₂C₁₂ + a₁₃C₁₃

Where Cᵢⱼ is the cofactor (signed minor).

**Example**: Find det[1  2  3]
                  [0  1  2]
                  [2  1  0]

det = 1(1(0) - 2(1)) - 2(0(0) - 2(2)) + 3(0(1) - 1(2))
    = 1(-2) - 2(-4) + 3(-2)
    = -2 + 8 - 6 = 0

### Properties of Determinants
1. det(AB) = det(A)det(B)
2. det(Aᵀ) = det(A)
3. det(cA) = cⁿdet(A) for n×n matrix
4. If two rows/columns are identical, det = 0
5. Swapping two rows/columns changes sign

## Cramer's Rule

### For 2×2 Systems
For system:
ax + by = e
cx + dy = f

x = det[e  b] / det[a  b]
       [f  d]       [c  d]

y = det[a  e] / det[a  b]
       [c  f]       [c  d]

**Example**: Solve 2x + 3y = 7
                x - y = 1

det = 2(-1) - 3(1) = -5

x = det[7  3] / (-5) = (7(-1) - 3(1)) / (-5) = -10 / (-5) = 2
       [1 -1]

y = det[2  7] / (-5) = (2(1) - 7(1)) / (-5) = -5 / (-5) = 1

### For 3×3 Systems
Similar process with 3×3 determinants.

## Matrix Inverses

### Definition
For square matrix A, the inverse A⁻¹ satisfies: AA⁻¹ = A⁻¹A = I

### Finding Inverse (2×2)
For A = [a  b]
        [c  d]

A⁻¹ = (1/det(A))[d  -b]
                [-c  a]

**Example**: Find inverse of A = [2  3]
                                [1 -1]

det(A) = 2(-1) - 3(1) = -5
A⁻¹ = (1/(-5))[-1  -3] = [1/5   3/5]
                          [-1    2]   [1/5  -2/5]

### Finding Inverse (3×3)
Use augmented matrix method:
1. Write [A | I]
2. Use row operations to get [I | A⁻¹]

## Applications

### Solving Systems of Equations
Matrix equation: Ax = b
Solution: x = A⁻¹b (if A is invertible)

**Example**: Solve system using matrices:
2x + 3y = 7
x - y = 1

Matrix form: [2  3][x] = [7]
             [1 -1][y]   [1]

Solution: [x] = [2  3]⁻¹[7] = [1/5   3/5][7] = [2]
          [y]   [1 -1]  [1]   [1/5  -2/5][1]   [1]

### Transformations
Matrices represent geometric transformations:
- **Rotation**: [cos θ  -sin θ]
               [sin θ   cos θ]
- **Scaling**: [sₓ  0 ]
               [0   sᵧ]
- **Reflection**: Various matrices

### Economics
- Input-output models
- Leontief models
- Economic equilibrium

## Practice Problems

### Problem 1
Find the determinant of [3  2]
                        [1  4]

**Solution**:
det = 3(4) - 2(1) = 12 - 2 = 10

### Problem 2
Use Cramer's rule to solve:
3x + 2y = 8
x + 4y = 6

**Solution**:
det = 3(4) - 2(1) = 10

x = det[8  2] / 10 = (8(4) - 2(6)) / 10 = 20 / 10 = 2
       [6  4]

y = det[3  8] / 10 = (3(6) - 8(1)) / 10 = 10 / 10 = 1

### Problem 3
Find the inverse of [1  2]
                   [3  4]

**Solution**:
det = 1(4) - 2(3) = -2
A⁻¹ = (1/(-2))[4  -2] = [-2   1]
               [-3   1]   [3/2  -1/2]

### Problem 4
Solve using matrix inverse:
x + 2y = 5
3x + 4y = 11

**Solution**:
From Problem 3: A⁻¹ = [-2   1]
                      [3/2  -1/2]

[x] = [-2   1][5] = [-2(5) + 1(11)] = [1]
[y]   [3/2  -1/2][11] [3/2(5) - 1/2(11)] = [2]

## Key Takeaways
- Matrices organize data in rectangular arrays
- Matrix operations follow specific rules
- Determinants provide important information about matrices
- Cramer's rule offers alternative method for solving systems
- Matrix inverses enable solving matrix equations
- Applications span many fields including economics and computer graphics

## Next Steps
This completes our College Algebra series. You now have comprehensive coverage of all major topics including functions, polynomials, exponential/logarithmic functions, systems, sequences, conic sections, complex numbers, and matrices. These concepts provide the foundation for advanced mathematics courses.
