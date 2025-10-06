# Linear Algebra Tutorial 04: Determinants

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the definition and properties of determinants
- Compute determinants using cofactor expansion
- Apply Cramer's rule to solve linear systems
- Understand geometric interpretation of determinants
- Use determinants in various applications

## Introduction to Determinants

### Definition
The determinant of a square matrix A, denoted det(A) or |A|, is a scalar value that encodes important information about the matrix.

### Notation
- det(A) or |A| for determinant of matrix A
- |a b| for 2×2 matrix determinant
- |c d|

## Determinants of Small Matrices

### 2×2 Matrix
For A = [a b]
        [c d]

det(A) = ad - bc

**Example**: A = [2 3]
                [1 -1]
det(A) = 2(-1) - 3(1) = -2 - 3 = -5

### 3×3 Matrix
For A = [a₁₁ a₁₂ a₁₃]
        [a₂₁ a₂₂ a₂₃]
        [a₃₁ a₃₂ a₃₃]

det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)

**Example**: A = [1 2 3]
                [0 1 2]
                [0 0 1]

det(A) = 1(1·1 - 2·0) - 2(0·1 - 2·0) + 3(0·0 - 1·0) = 1(1) - 2(0) + 3(0) = 1

## Properties of Determinants

### Basic Properties
1. **det(I) = 1** (identity matrix)
2. **det(Aᵀ) = det(A)** (transpose property)
3. **det(AB) = det(A)det(B)** (multiplication property)
4. **det(A⁻¹) = 1/det(A)** (inverse property)

### Row/Column Operations
1. **Row Swap**: det(A') = -det(A) (swapping two rows)
2. **Row Scaling**: det(A') = c·det(A) (multiplying row by c)
3. **Row Addition**: det(A') = det(A) (adding multiple of one row to another)

### Special Cases
- **Triangular Matrix**: det(A) = product of diagonal elements
- **Zero Row/Column**: det(A) = 0
- **Identical Rows/Columns**: det(A) = 0

## Cofactor Expansion

### Definition
The cofactor Cᵢⱼ of element aᵢⱼ is:
Cᵢⱼ = (-1)ⁱ⁺ʲ det(Mᵢⱼ)

Where Mᵢⱼ is the minor (submatrix obtained by deleting row i and column j).

### Cofactor Expansion Formula
det(A) = Σ(j=1 to n) aᵢⱼCᵢⱼ (expansion along row i)
det(A) = Σ(i=1 to n) aᵢⱼCᵢⱼ (expansion along column j)

### Example
Find det(A) for A = [2 1 3]
                    [0 2 1]
                    [1 0 2]

**Solution**: Expand along first row:
det(A) = 2·C₁₁ + 1·C₁₂ + 3·C₁₃

C₁₁ = (-1)¹⁺¹ det([2 1]) = 1(4-0) = 4
                    [0 2]

C₁₂ = (-1)¹⁺² det([0 1]) = -1(0-1) = 1
                    [1 2]

C₁₃ = (-1)¹⁺³ det([0 2]) = 1(0-2) = -2
                    [1 0]

det(A) = 2(4) + 1(1) + 3(-2) = 8 + 1 - 6 = 3

## Cramer's Rule

### Statement
For system Ax = b where A is n×n and det(A) ≠ 0:
xᵢ = det(Aᵢ)/det(A)

Where Aᵢ is matrix A with column i replaced by vector b.

### Example
Solve using Cramer's rule:
2x + 3y = 7
x - y = 1

**Solution**:
A = [2  3], b = [7]
    [1 -1]      [1]

det(A) = 2(-1) - 3(1) = -5

A₁ = [7  3], det(A₁) = 7(-1) - 3(1) = -10
     [1 -1]

A₂ = [2  7], det(A₂) = 2(1) - 7(1) = -5
     [1  1]

x = det(A₁)/det(A) = -10/(-5) = 2
y = det(A₂)/det(A) = -5/(-5) = 1

## Geometric Interpretation

### 2D Case
For 2×2 matrix A = [a b]
                  [c d]

|det(A)| = area of parallelogram formed by vectors [a,c] and [b,d]

### 3D Case
For 3×3 matrix A, |det(A)| = volume of parallelepiped formed by the three column vectors.

### Orientation
- det(A) > 0: vectors form right-handed system
- det(A) < 0: vectors form left-handed system
- det(A) = 0: vectors are coplanar (linearly dependent)

## Applications

### Matrix Invertibility
A square matrix A is invertible if and only if det(A) ≠ 0.

### Linear Independence
Vectors v₁, v₂, ..., vₙ are linearly independent if and only if det([v₁ v₂ ... vₙ]) ≠ 0.

### Area and Volume Calculations
Determinants provide efficient ways to calculate areas and volumes in geometry.

### Change of Variables in Integration
Jacobian determinant is used in multivariable calculus for coordinate transformations.

## Practice Problems

### Problem 1
Compute det(A) for A = [3 1 2]
                      [0 2 1]
                      [1 0 3]

**Solution**:
Using cofactor expansion along first row:
det(A) = 3·C₁₁ + 1·C₁₂ + 2·C₁₃

C₁₁ = det([2 1]) = 6-0 = 6
      [0 3]

C₁₂ = -det([0 1]) = -(0-1) = 1
       [1 3]

C₁₃ = det([0 2]) = 0-2 = -2
      [1 0]

det(A) = 3(6) + 1(1) + 2(-2) = 18 + 1 - 4 = 15

### Problem 2
Use Cramer's rule to solve:
x + 2y + z = 3
2x - y + 3z = 1
x + y - z = 2

**Solution**:
A = [1  2  1], b = [3]
    [2 -1  3]      [1]
    [1  1 -1]      [2]

det(A) = 1(-1·(-1) - 3·1) - 2(2·(-1) - 3·1) + 1(2·1 - (-1)·1)
       = 1(1-3) - 2(-2-3) + 1(2+1)
       = 1(-2) - 2(-5) + 1(3)
       = -2 + 10 + 3 = 11

A₁ = [3  2  1], det(A₁) = 3(-1·(-1) - 3·1) - 2(1·(-1) - 3·2) + 1(1·1 - (-1)·2)
     [1 -1  3]        = 3(-2) - 2(-7) + 1(3) = -6 + 14 + 3 = 11
     [2  1 -1]

A₂ = [1  3  1], det(A₂) = 1(1·(-1) - 3·2) - 3(2·(-1) - 3·1) + 1(2·2 - 1·1)
     [2  1  3]        = 1(-7) - 3(-5) + 1(3) = -7 + 15 + 3 = 11
     [1  2 -1]

A₃ = [1  2  3], det(A₃) = 1(-1·2 - 1·1) - 2(2·2 - 1·1) + 3(2·1 - (-1)·1)
     [2 -1  1]        = 1(-3) - 2(3) + 3(3) = -3 - 6 + 9 = 0
     [1  1  2]

x = det(A₁)/det(A) = 11/11 = 1
y = det(A₂)/det(A) = 11/11 = 1
z = det(A₃)/det(A) = 0/11 = 0

### Problem 3
Find the area of triangle with vertices (0,0), (3,1), (1,2).

**Solution**:
Area = (1/2)|det([3 1])| = (1/2)|3·2 - 1·1| = (1/2)|6-1| = (1/2)(5) = 5/2
            [1 2]

## Key Takeaways
- Determinants encode important matrix properties
- Cofactor expansion provides systematic computation method
- Cramer's rule offers alternative solution method for linear systems
- Determinants have geometric interpretations for area and volume
- Zero determinant indicates linear dependence or non-invertibility

## Next Steps
In the next tutorial, we'll explore eigenvalues and eigenvectors, learning how to find them and understand their significance in matrix analysis and applications.
