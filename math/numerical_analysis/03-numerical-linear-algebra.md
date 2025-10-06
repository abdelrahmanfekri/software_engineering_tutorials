# Numerical Analysis Tutorial 03: Numerical Linear Algebra

## Learning Objectives
By the end of this tutorial, you will be able to:
- Implement LU decomposition for solving linear systems
- Apply QR decomposition for least squares problems
- Compute eigenvalues and eigenvectors numerically
- Use Singular Value Decomposition (SVD) for matrix analysis
- Implement iterative methods for large systems
- Apply numerical linear algebra to machine learning problems

## Introduction to Numerical Linear Algebra

### Why Numerical Linear Algebra?
Numerical linear algebra is crucial for:
- **Machine Learning**: Training algorithms, feature extraction
- **Scientific Computing**: Solving differential equations
- **Computer Graphics**: Transformations, rendering
- **Optimization**: Quadratic programming, constrained optimization

### Key Problems
1. **Linear Systems**: Solve Ax = b
2. **Least Squares**: Minimize ||Ax - b||₂
3. **Eigenvalue Problems**: Find λ, v such that Av = λv
4. **Matrix Decompositions**: Factor A = BC for analysis

## LU Decomposition

### Definition
For a square matrix A, LU decomposition factors A as:
A = LU

Where:
- L is lower triangular with 1's on diagonal
- U is upper triangular

### Algorithm (Gaussian Elimination with Partial Pivoting)
```
1. Initialize L = I, U = A
2. For k = 1 to n-1:
   a. Find pivot: p = argmax|U(i,k)| for i ≥ k
   b. Swap rows k and p in U and L
   c. For i = k+1 to n:
      - Set L(i,k) = U(i,k)/U(k,k)
      - U(i,:) = U(i,:) - L(i,k) * U(k,:)
```

### Solving Linear Systems
Given Ax = b and A = LU:

1. **Forward substitution**: Solve Ly = b
2. **Backward substitution**: Solve Ux = y

### Example: LU Decomposition
A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]

After decomposition:
L = [[1, 0, 0], [0.5, 1, 0], [0, 2/3, 1]]
U = [[2, 1, 0], [0, 1.5, 1], [0, 0, 4/3]]

### Complexity
- **Decomposition**: O(n³/3) operations
- **Forward/Backward substitution**: O(n²) each
- **Total for m right-hand sides**: O(n³/3 + mn²)

### Stability Considerations
- **Partial pivoting**: Prevents division by small numbers
- **Complete pivoting**: More stable but expensive
- **Condition number**: κ(A) affects accuracy

## QR Decomposition

### Definition
For matrix A (m × n), QR decomposition factors A as:
A = QR

Where:
- Q is orthogonal (m × m): QᵀQ = I
- R is upper triangular (m × n)

### Gram-Schmidt Process
```
1. Initialize Q = [], R = []
2. For k = 1 to n:
   a. vₖ = aₖ (k-th column of A)
   b. For i = 1 to k-1:
      - R(i,k) = qᵢᵀaₖ
      - vₖ = vₖ - R(i,k)qᵢ
   c. R(k,k) = ||vₖ||
   d. qₖ = vₖ/R(k,k)
   e. Add qₖ to Q
```

### Householder Reflections
More numerically stable than Gram-Schmidt:

```
For k = 1 to min(m,n):
1. v = aₖ(k:m) - ||aₖ(k:m)||e₁
2. v = v/||v||
3. H = I - 2vvᵀ
4. Apply H to A(k:m, k:n)
5. R(k,k) = ||aₖ(k:m)||
```

### Least Squares Problems
Given overdetermined system Ax = b (m > n):

**Normal equations**: AᵀAx = Aᵀb (may be ill-conditioned)
**QR approach**: 
1. A = QR
2. QRx = b ⟹ Rx = Qᵀb
3. Solve upper triangular system

### Example: Least Squares with QR
A = [[1, 1], [1, 2], [1, 3]], b = [1, 2, 4]

QR decomposition gives:
Q = [[1/√3, -1/√2], [1/√3, 0], [1/√3, 1/√2]]
R = [[√3, 2√3], [0, √2]]

Solution: x = [0.5, 1.5]ᵀ

## Eigenvalue Computation

### Power Method
Find dominant eigenvalue and eigenvector:

```
1. Choose random vector x₀
2. For k = 1, 2, ...
   a. yₖ = Axₖ₋₁
   b. xₖ = yₖ/||yₖ||
   c. λₖ = xₖᵀAxₖ
3. Stop when ||xₖ - xₖ₋₁|| < tolerance
```

### QR Algorithm
Find all eigenvalues:

```
1. A₀ = A
2. For k = 0, 1, ...
   a. Compute QR decomposition: Aₖ = QₖRₖ
   b. Update: Aₖ₊₁ = RₖQₖ
   c. Eigenvalues converge to diagonal of Aₖ
```

### Example: Power Method
A = [[2, 1], [1, 2]], x₀ = [1, 0]ᵀ

```
k | xₖᵀ           | λₖ
--|---------------|-------
1 | [0.894, 0.447]| 2.618
2 | [0.949, 0.316]| 2.732
3 | [0.971, 0.239]| 2.828
```

Dominant eigenvalue: λ ≈ 3 (exact: λ = 3)

## Singular Value Decomposition (SVD)

### Definition
For matrix A (m × n), SVD factors A as:
A = UΣVᵀ

Where:
- U (m × m): orthogonal matrix of left singular vectors
- Σ (m × n): diagonal matrix of singular values
- V (n × n): orthogonal matrix of right singular vectors

### Properties
- **Singular values**: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
- **Rank**: rank(A) = number of nonzero singular values
- **Condition number**: κ(A) = σ₁/σᵣ

### Computing SVD
1. **Bidiagonalization**: A → B (upper bidiagonal)
2. **QR iterations**: Apply QR algorithm to BᵀB
3. **Recovery**: Extract U, Σ, V from iterations

### Applications

#### Low-Rank Approximation
For A = UΣVᵀ, best rank-k approximation:
Aₖ = UₖΣₖVₖᵀ

Where Uₖ, Vₖ contain first k columns, Σₖ first k singular values.

#### Principal Component Analysis (PCA)
Given data matrix X (n × p):
1. Center data: X̃ = X - μ
2. Compute SVD: X̃ = UΣVᵀ
3. Principal components: columns of V
4. Projected data: UΣ

#### Regularized Least Squares
Solve: min ||Ax - b||₂² + λ||x||₂²

Solution: x = V(Σ² + λI)⁻¹ΣUᵀb

## Iterative Methods

### Jacobi Method
For Ax = b, split A = D + L + U:

```
xₖ₊₁ = D⁻¹(b - (L + U)xₖ)
```

### Gauss-Seidel Method
```
xₖ₊₁ = (D + L)⁻¹(b - Uxₖ)
```

### Conjugate Gradient Method
For symmetric positive definite A:

```
1. Initialize x₀, r₀ = b - Ax₀, p₀ = r₀
2. For k = 0, 1, ...
   a. αₖ = (rₖᵀrₖ)/(pₖᵀApₖ)
   b. xₖ₊₁ = xₖ + αₖpₖ
   c. rₖ₊₁ = rₖ - αₖApₖ
   d. βₖ₊₁ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)
   e. pₖ₊₁ = rₖ₊₁ + βₖ₊₁pₖ
```

### Convergence
- **Jacobi**: Converges if A is strictly diagonally dominant
- **Gauss-Seidel**: Converges if A is positive definite
- **CG**: Converges in at most n iterations (exact arithmetic)

## Applications to Machine Learning

### Linear Regression
**Normal equations**: β = (XᵀX)⁻¹Xᵀy
**QR approach**: More numerically stable
**SVD approach**: Handles rank-deficient cases

### Principal Component Analysis
1. Compute SVD of centered data matrix
2. Principal components = right singular vectors
3. Explained variance = squared singular values

### Regularization
**Ridge regression**: (XᵀX + λI)β = Xᵀy
**Lasso**: Requires iterative methods (coordinate descent)

### Neural Networks
**Backpropagation**: Matrix-vector products
**Batch normalization**: Statistical computations
**Weight initialization**: SVD-based methods

## Numerical Considerations

### Condition Numbers
For linear system Ax = b:
||δx||/||x|| ≤ κ(A) ||δb||/||b||

Where κ(A) = ||A|| ||A⁻¹||

### Stability
- **LU**: Use partial pivoting
- **QR**: Householder more stable than Gram-Schmidt
- **SVD**: Most stable for rank-deficient matrices
- **Eigenvalues**: QR algorithm with shifts

### Sparsity
For sparse matrices (mostly zeros):
- **Storage**: Store only nonzero elements
- **Algorithms**: Avoid operations that create fill-in
- **Iterative methods**: Often preferred for large sparse systems

## Practice Problems

### Problem 1
Solve the linear system using LU decomposition:
[[2, 1, 0], [1, 2, 1], [0, 1, 2]]x = [1, 2, 3]ᵀ

**Solution**:
After LU decomposition and forward/backward substitution:
x = [0, 1, 1]ᵀ

### Problem 2
Find least squares solution using QR decomposition:
[[1, 1], [1, 2], [1, 3]]x = [1, 2, 4]ᵀ

**Solution**:
x = [0.5, 1.5]ᵀ

### Problem 3
Find dominant eigenvalue of A = [[3, 1], [1, 3]] using power method.

**Solution**:
Starting with x₀ = [1, 0]ᵀ:
After 5 iterations: λ ≈ 4 (exact: λ = 4)

## Key Takeaways
- LU decomposition is efficient for multiple right-hand sides
- QR decomposition is stable for least squares problems
- SVD is most general but computationally expensive
- Iterative methods are preferred for large sparse systems
- Condition numbers determine numerical stability
- Choose method based on problem structure and requirements

## Next Steps
In the next tutorial, we'll explore numerical integration methods including trapezoidal rule, Simpson's rule, and Gaussian quadrature.
