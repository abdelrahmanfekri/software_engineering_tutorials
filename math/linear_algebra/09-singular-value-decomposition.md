# Linear Algebra Tutorial 09: Singular Value Decomposition (SVD)

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the definition and significance of SVD
- Compute SVD for matrices
- Apply SVD to data compression
- Use SVD for Principal Component Analysis (PCA)
- Understand applications in machine learning and image processing

## Introduction to SVD

### Definition
Every matrix A (m×n) can be decomposed as:
A = UΣVᵀ

Where:
- U is m×m orthogonal matrix (left singular vectors)
- Σ is m×n diagonal matrix (singular values)
- V is n×n orthogonal matrix (right singular vectors)

### Key Properties
- Singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
- r = rank(A) is the number of non-zero singular values
- U and V contain orthonormal columns

### Example
Find SVD of A = [3 0]
                [0 2]

**Solution**:
AᵀA = [3 0][3 0] = [9 0]
      [0 2][0 2]   [0 4]

Eigenvalues of AᵀA: λ₁ = 9, λ₂ = 4
Singular values: σ₁ = 3, σ₂ = 2

For λ₁ = 9: (AᵀA - 9I)v = 0
[0  0][x] = [0] → x free, y = 0
[0 -5][y]   [0]
Right singular vector: v₁ = [1, 0]ᵀ

For λ₂ = 4: (AᵀA - 4I)v = 0
[5  0][x] = [0] → x = 0, y free
[0  0][y]   [0]
Right singular vector: v₂ = [0, 1]ᵀ

V = [1 0]
    [0 1]

U = AVΣ⁻¹ = [3 0][1 0][1/3  0 ] = [1 0]
            [0 2][0 1][0   1/2]   [0 1]

Σ = [3 0]
    [0 2]

Verification: UΣVᵀ = [1 0][3 0][1 0] = [3 0] = A ✓
                   [0 1][0 2][0 1]   [0 2]

## Computing SVD

### Method 1: Using AᵀA and AAᵀ
1. Compute AᵀA and find its eigenvalues and eigenvectors
2. Singular values are square roots of eigenvalues of AᵀA
3. Right singular vectors are eigenvectors of AᵀA
4. Left singular vectors: uᵢ = (1/σᵢ)Avᵢ

### Method 2: Direct Computation
For small matrices, use eigenvalue decomposition of AᵀA and AAᵀ.

### Example
Find SVD of A = [1 1]
                [0 1]

**Solution**:
AᵀA = [1 0][1 1] = [1 1]
      [1 1][0 1]   [1 2]

Characteristic polynomial: (1-λ)(2-λ) - 1 = λ² - 3λ + 1 = 0
Eigenvalues: λ₁ = (3+√5)/2, λ₂ = (3-√5)/2

Singular values: σ₁ = √((3+√5)/2), σ₂ = √((3-√5)/2)

For λ₁ = (3+√5)/2:
(AᵀA - λ₁I)v = 0 gives:
[1-(3+√5)/2    1    ][x] = [0]
[1           2-(3+√5)/2][y]   [0]

This gives: (-1-√5)/2 · x + y = 0
So: y = (1+√5)/2 · x

Normalized eigenvector: v₁ = [2/(1+√5), 1]ᵀ / ||[2/(1+√5), 1]||

Similarly for λ₂, we get v₂.

Then uᵢ = (1/σᵢ)Avᵢ for i = 1, 2.

## Low-Rank Approximation

### Best Rank-k Approximation
For matrix A with SVD A = UΣVᵀ, the best rank-k approximation is:
Aₖ = UₖΣₖVₖᵀ

Where Uₖ, Σₖ, Vₖ contain only the first k columns/rows.

### Error Bound
||A - Aₖ||₂ = σₖ₊₁

### Example
A = [1 2 3]
    [4 5 6]
    [7 8 9]

Best rank-1 approximation uses only the first singular value and corresponding vectors.

## Applications

### Data Compression
- Keep only largest singular values
- Compress images, signals, and datasets
- Trade-off between compression ratio and quality

### Principal Component Analysis (PCA)
- SVD of centered data matrix gives principal components
- First k principal components explain most variance
- Used for dimensionality reduction

### Image Processing
- Compress images by keeping only important singular values
- Denoise images by removing small singular values
- Applications in computer vision

### Machine Learning
- Latent Semantic Analysis (LSA)
- Collaborative filtering
- Feature extraction
- Regularization techniques

## Principal Component Analysis via SVD

### Process
1. Center the data matrix X (subtract column means)
2. Compute SVD: X = UΣVᵀ
3. Principal components are columns of V
4. Principal component scores are UΣ

### Example
Given data points: (1,2), (2,3), (3,4), (4,5)

**Step 1**: Center the data
Mean: (2.5, 3.5)
Centered data: (-1.5,-1.5), (-0.5,-0.5), (0.5,0.5), (1.5,1.5)

**Step 2**: Form data matrix
X = [-1.5 -1.5]
    [-0.5 -0.5]
    [ 0.5  0.5]
    [ 1.5  1.5]

**Step 3**: Compute SVD
XᵀX = [5  5]
      [5  5]

Eigenvalues: λ₁ = 10, λ₂ = 0
Singular values: σ₁ = √10, σ₂ = 0

Principal component: v₁ = [1/√2, 1/√2]ᵀ

## Practice Problems

### Problem 1
Find SVD of A = [2 0]
                [0 3]

**Solution**:
AᵀA = [4 0]
      [0 9]

Eigenvalues: λ₁ = 9, λ₂ = 4
Singular values: σ₁ = 3, σ₂ = 2

Eigenvectors of AᵀA:
For λ₁ = 9: v₁ = [0, 1]ᵀ
For λ₂ = 4: v₂ = [1, 0]ᵀ

V = [0 1]
    [1 0]

U = AVΣ⁻¹ = [2 0][0 1][1/3  0 ] = [0 1]
            [0 3][1 0][0   1/2]   [1 0]

Σ = [3 0]
    [0 2]

### Problem 2
Find rank-1 approximation of A = [1 2]
                                  [3 4]

**Solution**:
AᵀA = [10 14]
      [14 20]

Characteristic polynomial: (10-λ)(20-λ) - 196 = λ² - 30λ + 4 = 0
Eigenvalues: λ₁ = 15 + √221, λ₂ = 15 - √221

σ₁ = √(15 + √221), σ₂ = √(15 - √221)

For λ₁: v₁ = [1, (5+√221)/14]ᵀ (normalized)
For λ₂: v₂ = [1, (5-√221)/14]ᵀ (normalized)

Rank-1 approximation: A₁ = σ₁u₁v₁ᵀ

### Problem 3
Apply PCA to data: (0,1), (1,1), (2,2), (3,2)

**Solution**:
**Step 1**: Center data
Mean: (1.5, 1.5)
Centered: (-1.5,-0.5), (-0.5,-0.5), (0.5,0.5), (1.5,0.5)

**Step 2**: Data matrix
X = [-1.5 -0.5]
    [-0.5 -0.5]
    [ 0.5  0.5]
    [ 1.5  0.5]

**Step 3**: XᵀX = [5  1]
                 [1  1]

Eigenvalues: λ₁ = 3 + √5, λ₂ = 3 - √5

Principal component: v₁ = [1, (1+√5)/2]ᵀ (normalized)

## Matrix Completion

### Problem
Given partial matrix with missing entries, find complete matrix.

### SVD Approach
1. Fill missing entries with initial guess
2. Compute SVD
3. Keep only largest singular values
4. Reconstruct matrix
5. Repeat until convergence

## Key Takeaways
- SVD provides optimal low-rank approximations
- Singular values indicate importance of components
- SVD enables efficient data compression
- PCA is a special case of SVD
- SVD has wide applications in data science

## Next Steps
In the next tutorial, we'll explore quadratic forms, learning about their matrix representation, classification, and applications in optimization.
