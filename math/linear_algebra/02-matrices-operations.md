# Linear Algebra Tutorial 02: Matrices and Matrix Operations

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand matrix notation and terminology
- Perform matrix addition, subtraction, and multiplication
- Understand matrix properties and special matrices
- Find transpose and inverse of matrices
- Use elementary row operations
- Convert matrices to row echelon form

## Introduction to Matrices

### Definition
A matrix is a rectangular array of numbers arranged in rows and columns.

**Notation**: A = [aᵢⱼ] where aᵢⱼ is the element in row i, column j

**Example**: A = [2  3]
              [1 -1]
is a 2×2 matrix.

### Matrix Dimensions
An m×n matrix has m rows and n columns.

**Example**: [1  2  3]
            [4  5  6]
is a 2×3 matrix.

## Matrix Operations

### Matrix Addition
Add corresponding elements: (A + B)ᵢⱼ = aᵢⱼ + bᵢⱼ

**Example**: [2  3] + [1 -1] = [3  2]
            [1 -1]   [0  2]   [1  1]

### Matrix Subtraction
Subtract corresponding elements: (A - B)ᵢⱼ = aᵢⱼ - bᵢⱼ

**Example**: [2  3] - [1 -1] = [1  4]
            [1 -1]   [0  2]   [1 -3]

### Scalar Multiplication
Multiply each element by scalar: (cA)ᵢⱼ = c·aᵢⱼ

**Example**: 3[2  3] = [6  9]
            [1 -1]   [3 -3]

### Properties of Matrix Operations
1. **Commutativity**: A + B = B + A
2. **Associativity**: (A + B) + C = A + (B + C)
3. **Distributivity**: c(A + B) = cA + cB
4. **Identity**: A + O = A (where O is zero matrix)

## Matrix Multiplication

### Definition
For matrices A (m×n) and B (n×p), the product AB is an m×p matrix where:
(AB)ᵢⱼ = Σ(k=1 to n) aᵢₖbₖⱼ

### Step-by-Step Process
1. Ensure A has n columns and B has n rows
2. Element (i,j) of AB is dot product of row i of A and column j of B

**Example**: Multiply A = [2  3] and B = [1  2]
                        [1 -1]         [0  1]

AB = [2·1 + 3·0  2·2 + 3·1] = [2  7]
     [1·1 + (-1)·0  1·2 + (-1)·1] = [1  1]

### Properties of Matrix Multiplication
1. **Associativity**: (AB)C = A(BC)
2. **Distributivity**: A(B + C) = AB + AC
3. **NOT Commutative**: AB ≠ BA in general
4. **Identity**: AI = IA = A (where I is identity matrix)

## Special Matrices

### Zero Matrix
All elements are zero: O = [0]

### Identity Matrix
Square matrix with 1s on diagonal, 0s elsewhere:
I = [1  0  0]
    [0  1  0]
    [0  0  1]

### Diagonal Matrix
Non-zero elements only on main diagonal:
D = [2  0  0]
    [0 -1  0]
    [0  0  3]

### Triangular Matrices
- **Upper triangular**: Non-zero elements above main diagonal
- **Lower triangular**: Non-zero elements below main diagonal

## Matrix Transpose

### Definition
The transpose Aᵀ of matrix A is formed by interchanging rows and columns: (Aᵀ)ᵢⱼ = aⱼᵢ

**Example**: If A = [2  3  1]
                  [1 -1  0]
then Aᵀ = [2  1]
          [3 -1]
          [1  0]

### Properties of Transpose
1. (Aᵀ)ᵀ = A
2. (A + B)ᵀ = Aᵀ + Bᵀ
3. (cA)ᵀ = cAᵀ
4. (AB)ᵀ = BᵀAᵀ

## Matrix Inverse

### Definition
For square matrix A, the inverse A⁻¹ satisfies: AA⁻¹ = A⁻¹A = I

### Finding Inverse (2×2 Matrix)
For A = [a  b]
        [c  d]

A⁻¹ = (1/det(A))[d  -b]
                [-c  a]

Where det(A) = ad - bc

**Example**: Find inverse of A = [2  3]
                                [1 -1]

det(A) = 2(-1) - 3(1) = -2 - 3 = -5
A⁻¹ = (1/(-5))[-1  -3] = [1/5   3/5]
                          [-1    2]   [1/5  -2/5]

### Properties of Inverse
1. (A⁻¹)⁻¹ = A
2. (AB)⁻¹ = B⁻¹A⁻¹
3. (Aᵀ)⁻¹ = (A⁻¹)ᵀ

## Elementary Row Operations

### Types of Operations
1. **Row Swap**: Rᵢ ↔ Rⱼ
2. **Row Scaling**: Rᵢ → cRᵢ (c ≠ 0)
3. **Row Addition**: Rᵢ → Rᵢ + cRⱼ

### Elementary Matrices
Matrices that represent elementary row operations.

**Example**: E = [1  0  0] represents R₂ → R₂ + 2R₁
                [2  1  0]
                [0  0  1]

## Row Echelon Form

### Definition
A matrix is in row echelon form if:
1. All zero rows are at the bottom
2. First non-zero element in each row is 1 (leading 1)
3. Leading 1s move to the right as you go down

**Example**: [1  2  3  0]
            [0  1 -1  2]
            [0  0  0  1]
            [0  0  0  0]

### Reduced Row Echelon Form
Additional condition: Each leading 1 is the only non-zero element in its column.

**Example**: [1  0  0  2]
            [0  1  0 -1]
            [0  0  1  3]
            [0  0  0  0]

## Gaussian Elimination

### Process
1. Use elementary row operations to get row echelon form
2. Use back substitution to solve system

**Example**: Solve system using matrix:
2x + 3y = 7
x - y = 1

**Solution**:
Augmented matrix: [2  3 | 7]
                 [1 -1 | 1]

R₁ ↔ R₂: [1 -1 | 1]
         [2  3 | 7]

R₂ - 2R₁: [1 -1 | 1]
          [0  5 | 5]

(1/5)R₂: [1 -1 | 1]
         [0  1 | 1]

R₁ + R₂: [1  0 | 2]
         [0  1 | 1]

Solution: x = 2, y = 1

## Applications

### Solving Linear Systems
Matrix methods provide systematic approach to solving systems of equations.

### Computer Graphics
Matrices represent transformations (rotation, scaling, translation).

### Data Analysis
Matrices organize and manipulate large datasets.

## Practice Problems

### Problem 1
Multiply A = [1  2] and B = [3  1]
            [0 -1]         [2  0]

**Solution**:
AB = [1·3 + 2·2  1·1 + 2·0] = [7  1]
     [0·3 + (-1)·2  0·1 + (-1)·0] = [-2  0]

### Problem 2
Find the inverse of A = [3  2]
                        [1  1]

**Solution**:
det(A) = 3(1) - 2(1) = 1
A⁻¹ = (1/1)[1  -2] = [1  -2]
           [-1   3]   [-1   3]

### Problem 3
Use Gaussian elimination to solve:
x + 2y + z = 3
2x + 5y + 3z = 8
x + 4y + 2z = 7

**Solution**:
Augmented matrix: [1  2  1 | 3]
                 [2  5  3 | 8]
                 [1  4  2 | 7]

R₂ - 2R₁: [1  2  1 | 3]
          [0  1  1 | 2]
          [1  4  2 | 7]

R₃ - R₁: [1  2  1 | 3]
         [0  1  1 | 2]
         [0  2  1 | 4]

R₃ - 2R₂: [1  2  1 | 3]
          [0  1  1 | 2]
          [0  0 -1 | 0]

R₃ → -R₃: [1  2  1 | 3]
          [0  1  1 | 2]
          [0  0  1 | 0]

Back substitution: z = 0, y = 2, x = -1

## Key Takeaways
- Matrices are rectangular arrays of numbers
- Matrix operations follow specific rules
- Matrix multiplication is not commutative
- Elementary row operations preserve solution sets
- Row echelon form simplifies solving systems
- Matrix inverse provides solution method

## Next Steps
In the next tutorial, we'll explore systems of linear equations in detail, learning various solution methods and understanding consistency and inconsistency.
