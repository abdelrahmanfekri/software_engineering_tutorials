# Linear Algebra Tutorial 05: Eigenvalues and Eigenvectors

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand eigenvalues and eigenvectors conceptually
- Compute eigenvalues using the characteristic polynomial
- Find eigenvectors for given eigenvalues
- Understand diagonalization of matrices
- Apply eigenvalues/eigenvectors to stability analysis and PCA

## Introduction to Eigenvalues and Eigenvectors

### Definition
For a square matrix A, a non-zero vector v is an eigenvector if:
Av = λv

Where λ is the corresponding eigenvalue (scalar).

### Geometric Interpretation
- Eigenvectors are directions that don't change under the linear transformation
- Eigenvalues tell us how much vectors in those directions are stretched or compressed
- If λ > 1: stretching
- If 0 < λ < 1: compression
- If λ < 0: reflection and scaling

## Finding Eigenvalues

### Characteristic Polynomial
The characteristic polynomial of matrix A is:
p(λ) = det(A - λI)

The eigenvalues are the roots of p(λ) = 0.

### Example
Find eigenvalues of A = [3 1]
                        [0 2]

**Solution**:
A - λI = [3-λ  1  ]
         [0   2-λ]

det(A - λI) = (3-λ)(2-λ) - 1·0 = (3-λ)(2-λ) = 0

Roots: λ₁ = 3, λ₂ = 2

## Finding Eigenvectors

### Process
1. For each eigenvalue λᵢ, solve (A - λᵢI)v = 0
2. The solution space is the eigenspace E(λᵢ)
3. Any non-zero vector in E(λᵢ) is an eigenvector

### Example
Find eigenvectors for A = [3 1]
                          [0 2]

**Solution**:
For λ₁ = 3:
(A - 3I)v = [0  1][x] = [0]
            [0 -1][y]   [0]

This gives: y = 0, x is free
Eigenvector: v₁ = [1, 0]ᵀ (or any multiple)

For λ₂ = 2:
(A - 2I)v = [1  1][x] = [0]
            [0  0][y]   [0]

This gives: x + y = 0, so y = -x
Eigenvector: v₂ = [1, -1]ᵀ (or any multiple)

## Properties of Eigenvalues and Eigenvectors

### Basic Properties
1. **Sum of eigenvalues** = trace(A)
2. **Product of eigenvalues** = det(A)
3. **Eigenvalues of Aᵀ** = eigenvalues of A
4. **Eigenvalues of A⁻¹** = 1/λᵢ (if A is invertible)

### Special Cases
- **Diagonal matrices**: eigenvalues are diagonal elements
- **Triangular matrices**: eigenvalues are diagonal elements
- **Symmetric matrices**: all eigenvalues are real
- **Orthogonal matrices**: |λ| = 1 for all eigenvalues

## Diagonalization

### Definition
Matrix A is diagonalizable if there exists invertible matrix P such that:
P⁻¹AP = D

Where D is a diagonal matrix with eigenvalues on the diagonal.

### Process
1. Find eigenvalues λ₁, λ₂, ..., λₙ
2. Find corresponding eigenvectors v₁, v₂, ..., vₙ
3. Form P = [v₁ v₂ ... vₙ]
4. Form D = diag(λ₁, λ₂, ..., λₙ)
5. Verify P⁻¹AP = D

### Example
Diagonalize A = [3 1]
                [0 2]

**Solution**:
From previous example:
- λ₁ = 3, v₁ = [1, 0]ᵀ
- λ₂ = 2, v₂ = [1, -1]ᵀ

P = [1  1], P⁻¹ = [1  1]
    [0 -1]        [0 -1]

D = [3 0]
    [0 2]

Verification: P⁻¹AP = [1  1][3 1][1  1] = [3 0] = D
                      [0 -1][0 2][0 -1]   [0 2]

## Applications

### Stability Analysis
In dynamical systems x' = Ax:
- If all eigenvalues have negative real parts: system is stable
- If any eigenvalue has positive real part: system is unstable
- If eigenvalues are purely imaginary: system oscillates

### Principal Component Analysis (PCA)
- Eigenvalues of covariance matrix represent variance explained
- Eigenvectors represent principal directions
- Used for dimensionality reduction in data analysis

### Google PageRank
- Eigenvector of transition matrix represents page importance
- Largest eigenvalue corresponds to steady-state distribution

### Quantum Mechanics
- Eigenvalues represent possible energy levels
- Eigenvectors represent quantum states

## Practice Problems

### Problem 1
Find eigenvalues and eigenvectors of A = [2 1]
                                        [0 3]

**Solution**:
Characteristic polynomial: det(A - λI) = (2-λ)(3-λ) = 0
Eigenvalues: λ₁ = 2, λ₂ = 3

For λ₁ = 2:
(A - 2I)v = [0  1][x] = [0]
            [0  1][y]   [0]
Eigenvector: v₁ = [1, 0]ᵀ

For λ₂ = 3:
(A - 3I)v = [-1  1][x] = [0]
            [0   0][y]   [0]
Eigenvector: v₂ = [1, 1]ᵀ

### Problem 2
Diagonalize A = [4 1]
                [1 4]

**Solution**:
Characteristic polynomial: det(A - λI) = (4-λ)² - 1 = λ² - 8λ + 15 = 0
Eigenvalues: λ₁ = 5, λ₂ = 3

For λ₁ = 5:
(A - 5I)v = [-1  1][x] = [0]
            [1  -1][y]   [0]
Eigenvector: v₁ = [1, 1]ᵀ

For λ₂ = 3:
(A - 3I)v = [1  1][x] = [0]
            [1  1][y]   [0]
Eigenvector: v₂ = [1, -1]ᵀ

P = [1   1], P⁻¹ = (1/2)[1  1]
    [1  -1]              [1 -1]

D = [5 0]
    [0 3]

### Problem 3
Find eigenvalues of A = [1 2 0]
                        [0 2 0]
                        [0 0 3]

**Solution**:
Since A is upper triangular, eigenvalues are diagonal elements:
λ₁ = 1, λ₂ = 2, λ₃ = 3

## Spectral Theorem

### Statement
For symmetric matrix A, there exists orthogonal matrix Q such that:
QᵀAQ = D

Where D is diagonal with real eigenvalues.

### Implications
- Symmetric matrices are always diagonalizable
- Eigenvectors can be chosen to be orthonormal
- All eigenvalues are real

## Jordan Canonical Form

### When Diagonalization Fails
Not all matrices are diagonalizable. For non-diagonalizable matrices, we can find Jordan canonical form:

J = [λ₁  1   0]
    [0  λ₁   0]
    [0   0  λ₂]

### Example
A = [2 1] is not diagonalizable
    [0 2]

It has eigenvalue λ = 2 with algebraic multiplicity 2 but geometric multiplicity 1.

## Key Takeaways
- Eigenvalues and eigenvectors reveal fundamental matrix structure
- Characteristic polynomial provides systematic eigenvalue computation
- Diagonalization simplifies matrix operations
- Eigenvalues have important applications in stability and data analysis
- Not all matrices are diagonalizable

## Next Steps
In the next tutorial, we'll explore linear transformations, learning how matrices represent transformations and understanding kernel, image, and composition of transformations.
