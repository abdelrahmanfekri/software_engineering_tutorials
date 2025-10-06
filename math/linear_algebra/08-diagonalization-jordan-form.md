# Linear Algebra Tutorial 08: Diagonalization and Jordan Form

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand similarity transformations
- Determine when matrices are diagonalizable
- Find Jordan canonical form for non-diagonalizable matrices
- Apply diagonalization to solve differential equations
- Understand the spectral theorem for symmetric matrices

## Similarity Transformations

### Definition
Matrices A and B are similar if there exists invertible matrix P such that:
B = P⁻¹AP

### Properties of Similarity
1. **Reflexive**: A is similar to A
2. **Symmetric**: If A ~ B, then B ~ A
3. **Transitive**: If A ~ B and B ~ C, then A ~ C

### Invariants Under Similarity
- Determinant: det(A) = det(B)
- Trace: tr(A) = tr(B)
- Eigenvalues: A and B have same eigenvalues
- Rank: rank(A) = rank(B)

### Example
Show that A = [2 1] and B = [3 0] are similar
              [0 2]        [0 2]

**Solution**:
Find P such that B = P⁻¹AP, or equivalently PB = AP

Let P = [a b], then PB = AP gives:
        [c d]

[a b][3 0] = [2 1][a b]
[c d][0 2]   [0 2][c d]

[3a 2b] = [2a+c  2b+d]
[3c 2d]   [2c    2d]

This gives: 3a = 2a + c, 2b = 2b + d, 3c = 2c, 2d = 2d
So: a = c, d = 0, c = 0, which means a = c = 0

But P must be invertible, so this is impossible. A and B are NOT similar.

## Diagonalization

### Definition
Matrix A is diagonalizable if it's similar to a diagonal matrix D.

### Conditions for Diagonalization
A is diagonalizable if and only if:
1. A has n linearly independent eigenvectors (where n = size of A)
2. The geometric multiplicity equals algebraic multiplicity for each eigenvalue

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
Eigenvalues: λ₁ = 3, λ₂ = 2

For λ₁ = 3: (A - 3I)v = 0
[0  1][x] = [0] → y = 0, x free
[0 -1][y]   [0]
Eigenvector: v₁ = [1, 0]ᵀ

For λ₂ = 2: (A - 2I)v = 0
[1  1][x] = [0] → x + y = 0
[0  0][y]   [0]
Eigenvector: v₂ = [1, -1]ᵀ

P = [1   1], D = [3 0]
    [0  -1]      [0 2]

P⁻¹ = [1  1]
      [0 -1]

Verification: P⁻¹AP = [1  1][3 1][1   1] = [3 0] = D ✓
                      [0 -1][0 2][0  -1]   [0 2]

## Powers of Matrices

### Using Diagonalization
If A = PDP⁻¹, then Aᵏ = PDᵏP⁻¹

### Example
Compute A¹⁰⁰ for A = [3 1]
                     [0 2]

**Solution**:
From previous example: A = PDP⁻¹ where D = [3 0]
                                            [0 2]

A¹⁰⁰ = PD¹⁰⁰P⁻¹ = [1   1][3¹⁰⁰  0  ][1  1]
                   [0  -1][0    2¹⁰⁰][0 -1]

= [1   1][3¹⁰⁰  0  ]
  [0  -1][0    2¹⁰⁰]

= [3¹⁰⁰  2¹⁰⁰]
  [0     -2¹⁰⁰]

## Jordan Canonical Form

### When Diagonalization Fails
Not all matrices are diagonalizable. For non-diagonalizable matrices, we use Jordan form.

### Jordan Block
A Jordan block of size k with eigenvalue λ is:
Jₖ(λ) = [λ  1  0  ... 0]
        [0  λ  1  ... 0]
        [0  0  λ  ... 0]
        [⋮  ⋮  ⋮  ⋱  ⋮]
        [0  0  0  ... λ]

### Jordan Canonical Form
Every matrix is similar to a block diagonal matrix with Jordan blocks.

### Example
Find Jordan form of A = [2 1]
                        [0 2]

**Solution**:
Characteristic polynomial: (λ - 2)² = 0
Eigenvalue: λ = 2 with algebraic multiplicity 2

(A - 2I) = [0 1]
           [0 0]

Null space: (A - 2I)v = 0 gives y = 0, x free
Geometric multiplicity = 1 < algebraic multiplicity = 2

Since geometric multiplicity < algebraic multiplicity, A is not diagonalizable.

Jordan form: J = [2 1]
                 [0 2]

## Applications to Differential Equations

### System of Linear ODEs
Consider x' = Ax where A is diagonalizable.

If A = PDP⁻¹, then the solution is:
x(t) = Pe^(Dt)P⁻¹x₀

Where e^(Dt) = diag(e^(λ₁t), e^(λ₂t), ..., e^(λₙt))

### Example
Solve x' = Ax where A = [3 1] and x(0) = [1]
                        [0 2]           [1]

**Solution**:
From previous diagonalization: A = PDP⁻¹ where D = [3 0]
                                                   [0 2]

e^(Dt) = [e^(3t)  0    ]
         [0      e^(2t)]

x(t) = [1   1][e^(3t)  0    ][1  1][1]
       [0  -1][0      e^(2t)][0 -1][1]

= [1   1][e^(3t)  0    ][2]
  [0  -1][0      e^(2t)][-1]

= [1   1][2e^(3t)  ]
  [0  -1][-e^(2t)  ]

= [2e^(3t) - e^(2t)]
  [e^(2t)         ]

## Spectral Theorem

### Statement
For symmetric matrix A, there exists orthogonal matrix Q such that:
QᵀAQ = D

Where D is diagonal with real eigenvalues.

### Implications
- Symmetric matrices are always diagonalizable
- Eigenvectors can be chosen orthonormal
- All eigenvalues are real
- Qᵀ = Q⁻¹ (orthogonal matrix)

### Example
Diagonalize A = [2 1]
                [1 2]

**Solution**:
Characteristic polynomial: (2-λ)² - 1 = λ² - 4λ + 3 = 0
Eigenvalues: λ₁ = 3, λ₂ = 1

For λ₁ = 3: (A - 3I)v = 0
[-1  1][x] = [0] → -x + y = 0
[1  -1][y]   [0]
Eigenvector: v₁ = [1, 1]ᵀ, normalized: [1/√2, 1/√2]ᵀ

For λ₂ = 1: (A - I)v = 0
[1  1][x] = [0] → x + y = 0
[1  1][y]   [0]
Eigenvector: v₂ = [1, -1]ᵀ, normalized: [1/√2, -1/√2]ᵀ

Q = [1/√2   1/√2], D = [3 0]
    [1/√2  -1/√2]      [0 1]

Verification: QᵀAQ = [1/√2   1/√2][2 1][1/√2   1/√2] = [3 0] = D ✓
                    [1/√2  -1/√2][1 2][1/√2  -1/√2]   [0 1]

## Practice Problems

### Problem 1
Determine if A = [1 2] is diagonalizable
                [0 1]

**Solution**:
Characteristic polynomial: (1-λ)² = 0
Eigenvalue: λ = 1 with algebraic multiplicity 2

(A - I) = [0 2]
          [0 0]

Null space: (A - I)v = 0 gives 2y = 0, so y = 0, x free
Geometric multiplicity = 1 < algebraic multiplicity = 2

A is NOT diagonalizable.

### Problem 2
Diagonalize A = [4 1]
                [1 4]

**Solution**:
Characteristic polynomial: (4-λ)² - 1 = λ² - 8λ + 15 = 0
Eigenvalues: λ₁ = 5, λ₂ = 3

For λ₁ = 5: (A - 5I)v = 0
[-1  1][x] = [0] → -x + y = 0
[1  -1][y]   [0]
Eigenvector: v₁ = [1, 1]ᵀ

For λ₂ = 3: (A - 3I)v = 0
[1  1][x] = [0] → x + y = 0
[1  1][y]   [0]
Eigenvector: v₂ = [1, -1]ᵀ

P = [1   1], P⁻¹ = (1/2)[1  1]
    [1  -1]              [1 -1]

D = [5 0]
    [0 3]

### Problem 3
Find Jordan form of A = [3 1 0]
                        [0 3 1]
                        [0 0 3]

**Solution**:
Characteristic polynomial: (3-λ)³ = 0
Eigenvalue: λ = 3 with algebraic multiplicity 3

(A - 3I) = [0 1 0]
           [0 0 1]
           [0 0 0]

Null space: (A - 3I)v = 0 gives y = 0, z = 0, x free
Geometric multiplicity = 1 < algebraic multiplicity = 3

Jordan form: J = [3 1 0]
                 [0 3 1]
                 [0 0 3]

## Key Takeaways
- Similarity preserves important matrix properties
- Diagonalization requires sufficient eigenvectors
- Jordan form handles non-diagonalizable matrices
- Powers of matrices are easily computed via diagonalization
- Spectral theorem guarantees diagonalization for symmetric matrices

## Next Steps
In the next tutorial, we'll explore Singular Value Decomposition (SVD), learning about this powerful factorization and its applications in data compression and principal component analysis.
